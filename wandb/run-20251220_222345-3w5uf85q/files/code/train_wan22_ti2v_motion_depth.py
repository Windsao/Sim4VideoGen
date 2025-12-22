"""
Training script for WAN2.2-5B Text-to-Image-to-Video with Motion and Depth Heads.

Two-stage training:
- Stage 1: Train motion head + depth head only (backbone frozen)
- Stage 2: Train backbone (LoRA) + motion head + depth head together
"""

import torch
import torch.nn.functional as F
import os
import json
import argparse
import glob
from tqdm import tqdm

from typing import Optional, Tuple

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset,
    DataProcessingOperator,
)
from diffsynth.trainers.image_sequence_loader import LoadImageSequenceWithMotion
from diffsynth.models.wan_video_dit_motion import (
    MotionVectorHead, DepthHead,
    compute_motion_loss, compute_depth_loss
)
from diffsynth.models.spatiotemporal_depth_head import (
    SpatioTemporalDepthHead, SpatioTemporalDepthHeadSimple
)

from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _extract_state_dict(obj) -> dict:
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(f"Unsupported checkpoint format: {type(obj)}")


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_torch_state_dict(path: str, map_location) -> dict:
    obj = torch.load(path, map_location=map_location)
    state_dict = _extract_state_dict(obj)
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    return state_dict


def _expand_wan_dit_checkpoint_entry(entry: str) -> list[str]:
    """
    Expand a WAN DiT "checkpoint entry" into a list of weight files.

    ModelManager cannot load from a directory directly (unless it looks like a HF folder),
    but it *can* load from a list of files by merging state_dicts.
    """
    if not os.path.isdir(entry):
        return [entry]

    index_json = os.path.join(entry, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_json):
        with open(index_json, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        resolved = [os.path.join(entry, shard) for shard in shard_files]
        missing = [p for p in resolved if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Missing shard files referenced by {index_json}: {missing[:3]}")
        return resolved

    patterns = [
        os.path.join(entry, "diffusion_pytorch_model*.safetensors"),
        os.path.join(entry, "diffusion_pytorch_model*.bin"),
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    matches = sorted([p for p in matches if os.path.isfile(p) and not p.endswith(".index.json")])
    if matches:
        return matches

    # One-level fallback for releases that nest the actual weight files.
    subdirs = [os.path.join(entry, d) for d in os.listdir(entry) if os.path.isdir(os.path.join(entry, d))]
    for subdir in sorted(subdirs):
        try:
            sub_matches = _expand_wan_dit_checkpoint_entry(subdir)
        except Exception:
            continue
        if any(os.path.basename(p).startswith("diffusion_pytorch_model") for p in sub_matches):
            return sub_matches

    raise FileNotFoundError(
        f"Could not find DiT checkpoint files in directory: {entry}. "
        "Expected diffusion_pytorch_model*.safetensors (or an index json)."
    )


def normalize_model_paths_arg(model_paths: Optional[str]) -> Optional[str]:
    if model_paths is None:
        return None
    raw = json.loads(model_paths)
    normalized = []
    for item in raw:
        if isinstance(item, str) and os.path.isdir(item):
            normalized.append(_expand_wan_dit_checkpoint_entry(item))
        else:
            normalized.append(item)
    return json.dumps(normalized)


def summarize_model_paths(model_paths_json: Optional[str]) -> str:
    if model_paths_json is None:
        return "<none>"
    paths = json.loads(model_paths_json)
    parts = []
    for item in paths:
        if isinstance(item, list):
            if len(item) == 0:
                parts.append("[]")
            elif len(item) == 1:
                parts.append(f"[1 file] {item[0]}")
            else:
                parts.append(f"[{len(item)} files] {item[0]} ... {item[-1]}")
        else:
            parts.append(str(item))
    return " | ".join(parts)


def collate_keep_pil(batch):
    """
    PyTorch default_collate can't handle PIL.Image. We keep samples as-is.
    - batch_size == 1: return the single dict sample
    - batch_size > 1: return list[dict] and handle in training loop
    """
    if len(batch) == 1:
        return batch[0]
    return batch


class WAN22MotionDepthTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        # Motion-specific parameters
        motion_channels: int = 4,
        motion_loss_weight: float = 0.1,
        motion_loss_type: str = "mse",
        # Training mode
        training_mode: str = "heads_only",  # "heads_only", "lora", "both_last"
        # Depth-specific parameters
        depth_loss_weight: float = 0.1,
        depth_loss_type: str = "mse",
        # Warp loss parameters
        use_warp_loss: bool = False,
        warp_loss_weight: float = 0.1,
        warp_loss_type: str = "mse",
        # RGB warp loss parameters
        use_rgb_warp_loss: bool = False,
        rgb_warp_loss_weight: float = 0.1,
        rgb_warp_loss_type: str = "l1",
        rgb_warp_use_ssim: bool = False,
        rgb_warp_ssim_weight: float = 0.85,
        # Spatio-temporal depth head parameters
        use_spatiotemporal_depth: bool = False,
        spatiotemporal_depth_type: str = "simple",
        num_temporal_heads: int = 8,
        temporal_head_dim: int = 64,
        num_temporal_blocks: int = 2,
        temporal_pos_embed_type: str = "rope",
        # Checkpoint loading
        motion_head_checkpoint: str = None,
        depth_head_checkpoint: str = None,
        tokenizer_path: Optional[str] = None,
    ):
        super().__init__()

        # Load models
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )

        tokenizer_config = (
            ModelConfig(path=tokenizer_path)
            if tokenizer_path is not None
            else ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")
        )

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        if self.pipe.dit is None:
            raise RuntimeError(
                "Failed to load WAN DiT backbone (pipe.dit is None). "
                "This usually means your first --model_paths entry did not resolve to a valid DiT checkpoint. "
                "For WAN2.2 sharded weights, pass either the index folder that contains "
                "`diffusion_pytorch_model.safetensors.index.json` or a JSON list of shard files."
            )

        # Training mode
        self.training_mode = training_mode

        # Determine which heads to enable
        self.enable_motion = training_mode in ["lora", "both_last", "heads_only", "motion_only"]
        self.enable_depth = training_mode in ["lora", "both_last", "heads_only", "depth_only"]

        # Motion parameters
        self.motion_channels = motion_channels
        self.motion_loss_weight = motion_loss_weight if self.enable_motion else 0.0
        self.motion_loss_type = motion_loss_type

        # Depth parameters
        self.depth_loss_weight = depth_loss_weight if self.enable_depth else 0.0
        self.depth_loss_type = depth_loss_type

        # Warp loss parameters
        self.use_warp_loss = use_warp_loss and self.enable_depth
        self.warp_loss_weight = warp_loss_weight if self.use_warp_loss else 0.0
        self.warp_loss_type = warp_loss_type

        # RGB warp loss parameters
        self.use_rgb_warp_loss = use_rgb_warp_loss
        self.rgb_warp_loss_weight = rgb_warp_loss_weight if use_rgb_warp_loss else 0.0
        self.rgb_warp_loss_type = rgb_warp_loss_type
        self.rgb_warp_use_ssim = rgb_warp_use_ssim
        self.rgb_warp_ssim_weight = rgb_warp_ssim_weight

        # Spatio-temporal depth parameters
        self.use_spatiotemporal_depth = use_spatiotemporal_depth
        self.spatiotemporal_depth_type = spatiotemporal_depth_type
        self.num_temporal_heads = num_temporal_heads
        self.temporal_head_dim = temporal_head_dim
        self.num_temporal_blocks = num_temporal_blocks
        self.temporal_pos_embed_type = temporal_pos_embed_type

        # Initialize feature capture variables
        self._dit_features = None
        self._dit_timestep_embed = None  # time embedding passed into DiT head

        # Setup feature capture hook if either motion or depth is enabled
        if self.enable_motion or self.enable_depth:
            self._setup_feature_capture_hook()

        # Create and attach motion head if enabled
        if self.enable_motion:
            self._setup_motion_head(motion_head_checkpoint)

        # Create and attach depth head if enabled
        if self.enable_depth:
            self._setup_depth_head(depth_head_checkpoint)

        # Setup training mode (LoRA, freeze layers, etc.)
        self._setup_training_mode(
            trainable_models, lora_base_model, lora_target_modules,
            lora_rank, lora_checkpoint
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _setup_feature_capture_hook(self):
        """Capture DiT features right before the output head (keeps gradient flow)."""
        dit = self.pipe.dit

        def capture_features_hook(module, input, output):
            self._dit_features = input[0]
            self._dit_timestep_embed = input[1] if len(input) > 1 else None

        dit.head.register_forward_hook(capture_features_hook)

    def _setup_motion_head(self, checkpoint_path=None):
        """Create and attach motion vector head."""
        dit = self.pipe.dit
        dim = dit.dim
        patch_size = dit.patch_size
        eps = 1e-6

        self.motion_head = MotionVectorHead(
            dim=dim,
            motion_channels=self.motion_channels,
            patch_size=patch_size,
            eps=eps,
            output_scale=1.0,
        )

        dit_dtype = next(dit.parameters()).dtype
        self.motion_head = self.motion_head.to(dtype=dit_dtype, device=self.pipe.device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading motion head checkpoint from {checkpoint_path}")
            state_dict = load_torch_state_dict(checkpoint_path, map_location=self.pipe.device)
            load_result = self.motion_head.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise ValueError(
                    "Motion head checkpoint key mismatch. "
                    f"Missing: {load_result.missing_keys[:5]} "
                    f"Unexpected: {load_result.unexpected_keys[:5]}"
                )
            print(f"✓ Motion head checkpoint loaded ({len(state_dict)} tensors)")

        print(f"Motion head initialized with {self.motion_channels} channels")

    def _setup_depth_head(self, checkpoint_path=None):
        """Create and attach depth head."""
        dit = self.pipe.dit
        dim = dit.dim
        patch_size = dit.patch_size
        eps = 1e-6

        if self.use_spatiotemporal_depth:
            # Use spatio-temporal depth head
            if self.spatiotemporal_depth_type == "full":
                self.depth_head = SpatioTemporalDepthHead(
                    dim=dim,
                    patch_size=patch_size,
                    num_temporal_heads=self.num_temporal_heads,
                    temporal_head_dim=self.temporal_head_dim,
                    num_temporal_blocks=self.num_temporal_blocks,
                    pos_embed_type=self.temporal_pos_embed_type,
                    eps=eps,
                    output_scale=1.0,
                ).to(self.pipe.device)
            else:
                self.depth_head = SpatioTemporalDepthHeadSimple(
                    dim=dim,
                    patch_size=patch_size,
                    num_temporal_heads=self.num_temporal_heads,
                    temporal_head_dim=self.temporal_head_dim,
                    num_temporal_blocks=self.num_temporal_blocks,
                    pos_embed_type=self.temporal_pos_embed_type,
                    eps=eps,
                    output_scale=1.0,
                ).to(self.pipe.device)
            print(f"Spatio-temporal depth head initialized ({self.spatiotemporal_depth_type})")
        else:
            # Use standard depth head
            self.depth_head = DepthHead(
                dim=dim,
                patch_size=patch_size,
                eps=eps,
                output_scale=1.0,
            )
            self.depth_head = self.depth_head.to(device=self.pipe.device)
            print("Standard depth head initialized")

        dit_dtype = next(dit.parameters()).dtype
        self.depth_head = self.depth_head.to(dtype=dit_dtype, device=self.pipe.device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading depth head checkpoint from {checkpoint_path}")
            state_dict = load_torch_state_dict(checkpoint_path, map_location=self.pipe.device)
            load_result = self.depth_head.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                raise ValueError(
                    "Depth head checkpoint key mismatch. "
                    f"Missing: {load_result.missing_keys[:5]} "
                    f"Unexpected: {load_result.unexpected_keys[:5]}"
                )
            print(f"✓ Depth head checkpoint loaded ({len(state_dict)} tensors)")

    def _enforce_dit_lora_only_trainable(self):
        for name, param in self.pipe.dit.named_parameters():
            param.requires_grad = "lora" in name

    def _log_trainable_params_summary(self):
        if os.environ.get("RANK", "0") != "0":
            return
        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        non_lora_dit = [n for n in trainable_names if n.startswith("pipe.dit.") and "lora" not in n]
        print(f"[INFO] Trainable params: {len(trainable_names)} tensors")
        if non_lora_dit:
            print(f"[WARN] Non-LoRA DiT params are trainable (showing up to 10): {non_lora_dit[:10]}")
        else:
            print("[INFO] DiT trainable params are LoRA-only")

    def _setup_training_mode(
        self, trainable_models, lora_base_model,
        lora_target_modules, lora_rank, lora_checkpoint
    ):
        """Setup training mode: freeze/unfreeze layers, add LoRA, etc."""
        # Scheduler must be in training mode for diffusion training.
        self.pipe.scheduler.set_timesteps(1000, training=True)

        if self.training_mode == "heads_only":
            # Stage 1: Train only motion and depth heads, freeze everything else
            print("=" * 60)
            print("TRAINING MODE: Heads Only (Stage 1)")
            print("=" * 60)

            # Freeze everything in pipeline (DiT + encoders + VAE)
            self.pipe.freeze_except([])
            print("✗ Pipeline backbone: FROZEN")

            # Unfreeze motion head
            if self.enable_motion:
                self.motion_head.train()
                self.motion_head.requires_grad_(True)
                print("✓ Motion head: TRAINABLE")

            # Unfreeze depth head
            if self.enable_depth:
                self.depth_head.train()
                self.depth_head.requires_grad_(True)
                print("✓ Depth head: TRAINABLE")

            print("=" * 60)

        elif self.training_mode == "lora":
            # Stage 2: Train backbone with LoRA + motion/depth heads
            print("=" * 60)
            print("TRAINING MODE: LoRA + Heads (Stage 2)")
            print("=" * 60)

            # Setup LoRA for DiT
            self.switch_pipe_to_training_mode(
                self.pipe,
                trainable_models=trainable_models,
                lora_base_model=lora_base_model,
                lora_target_modules=lora_target_modules,
                lora_rank=lora_rank,
                lora_checkpoint=lora_checkpoint,
                enable_fp8_training=False,
            )

            # Default Stage 2 behavior: train LoRA adapters only (not full DiT).
            if trainable_models is None:
                self._enforce_dit_lora_only_trainable()

            # Ensure motion head is trainable
            if self.enable_motion:
                self.motion_head.train()
                self.motion_head.requires_grad_(True)
                print("✓ Motion head: TRAINABLE")

            # Ensure depth head is trainable
            if self.enable_depth:
                self.depth_head.train()
                self.depth_head.requires_grad_(True)
                print("✓ Depth head: TRAINABLE")

            self._log_trainable_params_summary()
            print(f"✓ DiT backbone: LoRA (rank={lora_rank})")
            print("=" * 60)

        elif self.training_mode == "both_last":
            # Alternative: Train heads + last DiT layer only
            print("=" * 60)
            print("TRAINING MODE: Heads + Last DiT Layer")
            print("=" * 60)

            self.pipe.freeze_except([])

            # Unfreeze last DiT block
            if hasattr(self.pipe.dit, 'blocks'):
                self.pipe.dit.blocks[-1].train()
                self.pipe.dit.blocks[-1].requires_grad_(True)
            elif hasattr(self.pipe.dit, 'transformer_blocks'):
                self.pipe.dit.transformer_blocks[-1].train()
                self.pipe.dit.transformer_blocks[-1].requires_grad_(True)

            # Unfreeze heads
            if self.enable_motion:
                self.motion_head.train()
                self.motion_head.requires_grad_(True)
                print("✓ Motion head: TRAINABLE")

            if self.enable_depth:
                self.depth_head.train()
                self.depth_head.requires_grad_(True)
                print("✓ Depth head: TRAINABLE")

            print("✓ DiT last layer: TRAINABLE")
            print("=" * 60)

    def _get_grid_size(self, latents: torch.Tensor) -> Tuple[int, int, int]:
        patch_size = self.pipe.dit.patch_size
        f = latents.shape[2] // patch_size[0]
        h = latents.shape[3] // patch_size[1]
        w = latents.shape[4] // patch_size[2]
        return f, h, w

    def _unpatchify_head_output(
        self,
        head_output: torch.Tensor,
        grid_size: Tuple[int, int, int],
        channels: int,
    ) -> torch.Tensor:
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size
        from einops import rearrange
        return rearrange(
            head_output,
            'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f,
            h=h,
            w=w,
            x=patch_size[0],
            y=patch_size[1],
            z=patch_size[2],
            c=channels,
        )

    def compute_motion_prediction(self, features: torch.Tensor, t_embed: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        motion_patchified = self.motion_head(features, t_embed)
        return self._unpatchify_head_output(motion_patchified, grid_size, self.motion_channels)

    def compute_depth_prediction(self, features: torch.Tensor, t_embed: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        if self.use_spatiotemporal_depth:
            if self.spatiotemporal_depth_type == "full":
                depth_pred, _ = self.depth_head(features, t_embed, grid_size)
                return depth_pred
            depth_patchified, _ = self.depth_head(features, t_embed, grid_size)
            return self.depth_head.unpatchify(depth_patchified, grid_size)

        depth_patchified = self.depth_head(features, t_embed)
        return self._unpatchify_head_output(depth_patchified, grid_size, 1)

    def forward_preprocess(self, data):
        """Preprocess data for forward pass."""
        video_data = data["video"]

        if isinstance(video_data, dict):
            video_frames = video_data.get("video", None)
            motion_vectors = video_data.get("motion_vectors", None)
            depth_maps = video_data.get("depth_maps", None)
            if video_frames is None:
                raise ValueError(f"Expected 'video' key in data['video'] dict, got keys: {list(video_data.keys())}")
        elif isinstance(video_data, list):
            video_frames = video_data
            motion_vectors = data.get("motion_vectors", None)
            depth_maps = data.get("depth_maps", None)
        else:
            raise ValueError(f"Unexpected data['video'] type: {type(video_data)}. Expected dict or list.")

        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            "input_video": video_frames,
            "height": video_frames[0].size[1],
            "width": video_frames[0].size[0],
            "num_frames": len(video_frames),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs (input_image for TI2V)
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = video_frames[0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = video_frames[-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Store motion and depth ground truth if available
        if motion_vectors is not None:
            inputs_shared["target_motion_vectors"] = motion_vectors
        if depth_maps is not None:
            inputs_shared["target_depth_maps"] = depth_maps

        # Pipeline units will automatically process input parameters
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        """Forward pass with motion and depth losses."""
        if inputs is None:
            inputs = self.forward_preprocess(data)

        # Extract targets
        target_motion = inputs.pop("target_motion_vectors", None)
        target_depth = inputs.pop("target_depth_maps", None)

        # Standard training loss computation (mirrors train_wan_with_motion.py)
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)
        timestep = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,)).to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )

        inputs["latents"] = self.pipe.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.pipe.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        # Forward through DiT (also populates feature hook)
        self._dit_features = None
        self._dit_timestep_embed = None
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        noise_pred = self.pipe.model_fn(**models, **inputs, timestep=timestep)

        noise_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        noise_loss_weighted = noise_loss * self.pipe.scheduler.training_weight(timestep)

        loss_dict = {
            "noise_loss": noise_loss.detach().item(),
            "noise_loss_weighted": noise_loss_weighted.detach().item(),
            "timestep": timestep.item(),
        }

        include_noise_loss = self.training_mode != "heads_only"
        total_loss = noise_loss_weighted if include_noise_loss else None

        # Compute motion loss
        if self.enable_motion and self._dit_features is not None and target_motion is not None:
            latents = inputs["latents"]
            grid_size = self._get_grid_size(latents)
            t_embed = self._dit_timestep_embed

            if t_embed is None:
                raise RuntimeError("Failed to capture DiT timestep embedding from head hook.")

            motion_pred = self.compute_motion_prediction(self._dit_features, t_embed, grid_size)

            # Align shapes
            target_motion = target_motion.to(device=motion_pred.device, dtype=motion_pred.dtype)
            if target_motion.dim() == 4:
                target_motion = target_motion.unsqueeze(0)
            if target_motion.shape[2] == motion_pred.shape[2] - 1:
                pad = torch.zeros_like(target_motion[:, :, :1])
                target_motion = torch.cat([target_motion, pad], dim=2)
            if target_motion.shape[2:] != motion_pred.shape[2:]:
                target_motion = F.interpolate(target_motion, size=motion_pred.shape[2:], mode="trilinear", align_corners=False)

            motion_loss = compute_motion_loss(
                motion_pred, target_motion, loss_type=self.motion_loss_type
            )

            loss_dict["motion_loss"] = motion_loss.detach().item()
            total_loss = motion_loss * self.motion_loss_weight if total_loss is None else total_loss + self.motion_loss_weight * motion_loss

        # Compute depth loss
        if self.enable_depth and self._dit_features is not None and target_depth is not None:
            latents = inputs["latents"]
            grid_size = self._get_grid_size(latents)
            t_embed = self._dit_timestep_embed

            if t_embed is None:
                raise RuntimeError("Failed to capture DiT timestep embedding from head hook.")

            depth_pred = self.compute_depth_prediction(self._dit_features, t_embed, grid_size)

            # Align shapes
            target_depth = target_depth.to(device=depth_pred.device, dtype=depth_pred.dtype)
            if target_depth.dim() == 4:
                target_depth = target_depth.unsqueeze(0)
            if target_depth.shape[2:] != depth_pred.shape[2:]:
                target_depth = F.interpolate(target_depth, size=depth_pred.shape[2:], mode="trilinear", align_corners=False)

            depth_loss = compute_depth_loss(
                depth_pred, target_depth, loss_type=self.depth_loss_type
            )

            loss_dict["depth_loss"] = depth_loss.detach().item()
            total_loss = depth_loss * self.depth_loss_weight if total_loss is None else total_loss + self.depth_loss_weight * depth_loss

        if total_loss is None:
            total_loss = torch.zeros((), device=self.pipe.device, dtype=self.pipe.torch_dtype, requires_grad=True)
        loss_dict["total_loss"] = total_loss.detach().item()

        return total_loss, loss_dict


class LoadImageSequenceWithMotionWrapper(DataProcessingOperator):
    """
    Wrapper that integrates LoadImageSequenceWithMotion with the data pipeline.

    This handles the directory structure where each sample is a directory
    containing rgb/, motion_vectors/, and distance_to_camera/ subdirectories.
    """

    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        motion_channels=4,
        normalize_motion=False,
        motion_scale=1.0,
        depth_scale=1.0,
        normalize_depth=False,
        load_depth=True,
    ):
        self.loader = LoadImageSequenceWithMotion(
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            frame_processor=frame_processor,
            image_pattern="*.png",
            motion_pattern="*.npy",
            depth_pattern="*.npy",
            rgb_subdir="rgb",
            motion_subdir="motion_vectors",
            depth_subdir="distance_to_camera",
            motion_channels=motion_channels,
            normalize_motion=normalize_motion,
            motion_scale=motion_scale,
            depth_scale=depth_scale,
            normalize_depth=normalize_depth,
            load_depth=load_depth,
        )

    def __call__(self, data: str):
        return self.loader(data)


def create_motion_data_operator(
    base_path="",
    max_pixels=1920 * 1080,
    height=None,
    width=None,
    height_division_factor=16,
    width_division_factor=16,
    num_frames=81,
    time_division_factor=4,
    time_division_remainder=1,
    motion_channels=4,
    normalize_motion=False,
    motion_scale=1.0,
    depth_scale=1.0,
    normalize_depth=False,
    load_depth=True,
):
    """
    Create a data operator for loading image sequences with motion vectors and depth maps.
    """
    from diffsynth.trainers.unified_dataset import (
        RouteByType,
        ToAbsolutePath,
        ImageCropAndResize,
    )

    frame_processor = ImageCropAndResize(
        height, width, max_pixels, height_division_factor, width_division_factor
    )

    return RouteByType(
        operator_map=[
            (
                str,
                ToAbsolutePath(base_path)
                >> LoadImageSequenceWithMotionWrapper(
                    num_frames=num_frames,
                    time_division_factor=time_division_factor,
                    time_division_remainder=time_division_remainder,
                    frame_processor=frame_processor,
                    motion_channels=motion_channels,
                    normalize_motion=normalize_motion,
                    motion_scale=motion_scale,
                    depth_scale=depth_scale,
                    normalize_depth=normalize_depth,
                    load_depth=load_depth,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)

    # Model parameters (provide either model_paths or model_id_with_origin_paths)
    parser.add_argument("--model_paths", type=str, default=None,
                       help="JSON array of model paths, e.g., '[\"path1\", \"path2\"]'")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                       help="Comma-separated model_id:pattern pairs")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Local path to UMT5 tokenizer files (avoids online download).")
    parser.add_argument("--output_path", type=str, required=True)

    # Training mode
    parser.add_argument("--training_mode", type=str, default="heads_only",
                       choices=["heads_only", "lora", "both_last"])

    # LoRA parameters (for stage 2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_checkpoint", type=str, default=None)

    # Checkpoint loading
    parser.add_argument("--motion_head_checkpoint", type=str, default=None)
    parser.add_argument("--depth_head_checkpoint", type=str, default=None)

    # Loss weights
    parser.add_argument("--motion_channels", type=int, default=4)
    parser.add_argument("--motion_loss_weight", type=float, default=0.1)
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)
    parser.add_argument("--motion_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--depth_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="wan22-ti2v-motion-depth")

    # Spatio-temporal depth
    parser.add_argument("--use_spatiotemporal_depth", action="store_true")
    parser.add_argument("--spatiotemporal_depth_type", type=str, default="simple",
                       choices=["simple", "full"])

    args = parser.parse_args()

    args.model_paths = normalize_model_paths_arg(args.model_paths)
    if os.environ.get("RANK", "0") == "0":
        print(f"[INFO] Resolved --model_paths: {summarize_model_paths(args.model_paths)}")

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # Initialize wandb
    if args.use_wandb and accelerator.is_main_process:
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed, but --use_wandb was provided.")
        wandb.init(project=args.wandb_project, config=vars(args))

    # Create dataset with motion and depth loading
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=["video"],
        main_data_operator=create_motion_data_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.height * args.width,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            motion_channels=args.motion_channels,
            normalize_motion=False,
            motion_scale=1.0,
            depth_scale=1.0,
            normalize_depth=False,
            load_depth=True,
        ),
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_keep_pil,
    )

    # Create model
    model = WAN22MotionDepthTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        training_mode=args.training_mode,
        trainable_models=None,
        lora_base_model="dit" if args.training_mode == "lora" else None,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        extra_inputs="input_image",
        tokenizer_path=args.tokenizer_path,
        motion_channels=args.motion_channels,
        motion_loss_weight=args.motion_loss_weight,
        motion_loss_type=args.motion_loss_type,
        depth_loss_weight=args.depth_loss_weight,
        depth_loss_type=args.depth_loss_type,
        motion_head_checkpoint=args.motion_head_checkpoint,
        depth_head_checkpoint=args.depth_head_checkpoint,
        use_spatiotemporal_depth=args.use_spatiotemporal_depth,
        spatiotemporal_depth_type=args.spatiotemporal_depth_type,
    )

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if os.environ.get("RANK", "0") == "0":
        print(f"[INFO] Optimizer params: {len(trainable_params)} tensors (requires_grad=True)")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    global_step = 0
    os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                           disable=not accelerator.is_main_process)

        for batch in progress_bar:
            with accelerator.accumulate(model):
                if isinstance(batch, list):
                    # batch_size > 1: per-sample forward, then mean.
                    losses = []
                    loss_dict_sum = {}
                    for sample in batch:
                        loss_i, loss_dict_i = model(sample)
                        losses.append(loss_i)
                        for k, v in loss_dict_i.items():
                            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + float(v)
                    loss = torch.stack(losses).mean()
                    loss_dict = {k: v / len(batch) for k, v in loss_dict_sum.items()}
                else:
                    loss, loss_dict = model(batch)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            # Logging
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "noise": f"{loss_dict['noise_loss']:.4f}",
                    "motion": f"{loss_dict.get('motion_loss', 0):.4f}",
                    "depth": f"{loss_dict.get('depth_loss', 0):.4f}",
                })

                if args.use_wandb:
                    wandb.log({
                        "train/total_loss": loss_dict["total_loss"],
                        "train/noise_loss": loss_dict["noise_loss"],
                        "train/motion_loss": loss_dict.get("motion_loss", 0),
                        "train/depth_loss": loss_dict.get("depth_loss", 0),
                        "global_step": global_step,
                    })

            # Save checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)

                # Save motion head
                if unwrapped.enable_motion:
                    torch.save(
                        unwrapped.motion_head.state_dict(),
                        os.path.join(save_path, "motion_head.pth")
                    )

                # Save depth head
                if unwrapped.enable_depth:
                    torch.save(
                        unwrapped.depth_head.state_dict(),
                        os.path.join(save_path, "depth_head.pth")
                    )

                # Save LoRA if in stage 2
                if args.training_mode == "lora":
                    # Save LoRA weights
                    lora_state_dict = {}
                    for name, param in unwrapped.pipe.dit.named_parameters():
                        if "lora" in name and param.requires_grad:
                            lora_state_dict[name] = param.cpu()
                    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pth"))

                print(f"Checkpoint saved to {save_path}")

    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_path, "final")
        os.makedirs(final_path, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)

        if unwrapped.enable_motion:
            torch.save(
                unwrapped.motion_head.state_dict(),
                os.path.join(final_path, "motion_head.pth")
            )

        if unwrapped.enable_depth:
            torch.save(
                unwrapped.depth_head.state_dict(),
                os.path.join(final_path, "depth_head.pth")
            )

        if args.training_mode == "lora":
            lora_state_dict = {}
            for name, param in unwrapped.pipe.dit.named_parameters():
                if "lora" in name and param.requires_grad:
                    lora_state_dict[name] = param.cpu()
            torch.save(lora_state_dict, os.path.join(final_path, "lora_weights.pth"))

        print(f"Final checkpoint saved to {final_path}")

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
