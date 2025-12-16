#!/usr/bin/env python3
"""
Fine-tuning script for WAN model with LoRA + motion + depth + RGB losses.

This script loads a checkpoint from train_wan_with_motion.py (which has trained
motion and depth heads) and continues fine-tuning with LoRA while training all
three components simultaneously:
1. RGB/Noise prediction loss (standard diffusion denoising via LoRA)
2. Motion vector prediction loss (motion head)
3. Depth prediction loss (depth head)

Usage:
    accelerate launch train_wan_finetune_all.py \
        --dataset_base_path /path/to/dataset \
        --dataset_metadata_path data/metadata.csv \
        --checkpoint_path /path/to/checkpoint/step-XXX.safetensors \
        --motion_head_checkpoint /path/to/motion_head.pt \
        --depth_head_checkpoint /path/to/depth_head.pt \
        --height 480 \
        --width 832 \
        --num_frames 81 \
        --model_paths '["/path/to/model1.safetensors", ...]' \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --output_path ./output/model_lora_all \
        --lora_base_model "dit" \
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
        --lora_rank 32 \
        --motion_loss_weight 0.1 \
        --depth_loss_weight 0.1 \
        --noise_loss_weight 1.0
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from typing import Optional, Dict, Any
from tqdm import tqdm

from safetensors.torch import save_file, load_file
from diffsynth import load_state_dict


# ============================================
# Timestep Sampling Strategies
# ============================================

def sample_timestep_uniform(min_t: int, max_t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard uniform sampling."""
    return torch.randint(min_t, max_t, (1,)).to(dtype=dtype, device=device)


def sample_timestep_high_noise_bias(
    min_t: int, max_t: int, device: torch.device, dtype: torch.dtype,
    bias_strength: float = 2.0
) -> torch.Tensor:
    """
    Sample with bias towards high noise (high timesteps).

    Uses power transformation: t = t_uniform^(1/bias_strength)
    - bias_strength > 1: more samples from high timesteps (high noise)
    - bias_strength = 1: uniform
    - bias_strength < 1: more samples from low timesteps (low noise)
    """
    u = torch.rand(1).item()  # Uniform [0, 1)
    # Transform to bias towards high values
    t_normalized = u ** (1.0 / bias_strength)
    timestep = int(min_t + t_normalized * (max_t - min_t))
    timestep = min(max(timestep, min_t), max_t - 1)
    return torch.tensor([timestep], dtype=dtype, device=device)


def sample_timestep_logit_normal(
    min_t: int, max_t: int, device: torch.device, dtype: torch.dtype,
    mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """
    Logit-normal sampling (used in SD3/Flux).

    Samples from a logit-normal distribution which can be centered or biased.
    - mean > 0: bias towards high noise
    - mean < 0: bias towards low noise
    - mean = 0: centered distribution
    """
    # Sample from normal distribution
    normal_sample = torch.randn(1).item() * std + mean
    # Apply sigmoid to map to [0, 1]
    t_normalized = 1.0 / (1.0 + np.exp(-normal_sample))
    timestep = int(min_t + t_normalized * (max_t - min_t))
    timestep = min(max(timestep, min_t), max_t - 1)
    return torch.tensor([timestep], dtype=dtype, device=device)


def sample_timestep_cubic_high(min_t: int, max_t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cubic transformation biased towards high noise (t^(1/3))."""
    u = torch.rand(1).item()
    t_normalized = u ** (1.0 / 3.0)
    timestep = int(min_t + t_normalized * (max_t - min_t))
    timestep = min(max(timestep, min_t), max_t - 1)
    return torch.tensor([timestep], dtype=dtype, device=device)


def sample_timestep_linear_high(
    min_t: int, max_t: int, device: torch.device, dtype: torch.dtype,
    min_prob_ratio: float = 0.2
) -> torch.Tensor:
    """
    Linear probability distribution biased towards high timesteps.

    P(t) increases linearly from min_prob_ratio at t=0 to 1.0 at t=max.
    """
    # Sample using inverse CDF of linear distribution
    u = torch.rand(1).item()
    # For linear CDF: F(x) = min_prob_ratio * x + (1 - min_prob_ratio) * x^2 / 2
    # Normalized so F(1) = 1
    # Inverse: x = (-min_prob_ratio + sqrt(min_prob_ratio^2 + 2*(1-min_prob_ratio)*u)) / (1 - min_prob_ratio)
    a = min_prob_ratio
    b = 1.0 - min_prob_ratio
    if b < 1e-6:
        t_normalized = u
    else:
        t_normalized = (-a + np.sqrt(a * a + 2 * b * u * (a + b / 2))) / b
    t_normalized = min(max(t_normalized, 0.0), 1.0)
    timestep = int(min_t + t_normalized * (max_t - min_t))
    timestep = min(max(timestep, min_t), max_t - 1)
    return torch.tensor([timestep], dtype=dtype, device=device)


def sample_timestep_beta_high(
    min_t: int, max_t: int, device: torch.device, dtype: torch.dtype,
    alpha: float = 1.0, beta: float = 0.5
) -> torch.Tensor:
    """
    Beta distribution sampling.

    - alpha=1, beta=0.5: strong bias towards high timesteps
    - alpha=0.5, beta=1: strong bias towards low timesteps
    - alpha=1, beta=1: uniform
    """
    t_normalized = np.random.beta(alpha, beta)
    timestep = int(min_t + t_normalized * (max_t - min_t))
    timestep = min(max(timestep, min_t), max_t - 1)
    return torch.tensor([timestep], dtype=dtype, device=device)


def sample_timestep_truncated_high(
    min_t: int, max_t: int, device: torch.device, dtype: torch.dtype,
    high_ratio: float = 0.7
) -> torch.Tensor:
    """
    Truncated sampling that samples more from high timesteps.

    With probability high_ratio, sample from top 50% of timesteps.
    With probability (1-high_ratio), sample uniformly.
    """
    if torch.rand(1).item() < high_ratio:
        # Sample from top 50%
        mid_t = (min_t + max_t) // 2
        timestep = torch.randint(mid_t, max_t, (1,)).item()
    else:
        # Sample uniformly
        timestep = torch.randint(min_t, max_t, (1,)).item()
    return torch.tensor([timestep], dtype=dtype, device=device)


TIMESTEP_SAMPLERS = {
    "uniform": sample_timestep_uniform,
    "high_noise_bias": sample_timestep_high_noise_bias,
    "logit_normal": sample_timestep_logit_normal,
    "logit_normal_high": lambda min_t, max_t, device, dtype: sample_timestep_logit_normal(
        min_t, max_t, device, dtype, mean=0.5, std=1.0
    ),
    "cubic_high": sample_timestep_cubic_high,
    "linear_high": sample_timestep_linear_high,
    "beta_high": sample_timestep_beta_high,
    "truncated_high": sample_timestep_truncated_high,
}
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset,
    RouteByType,
    ToAbsolutePath,
    ImageCropAndResize,
    SequencialProcess,
    DataProcessingOperator,
)
from diffsynth.trainers.image_sequence_loader import LoadImageSequenceWithMotion
from diffsynth.models.wan_video_dit_motion import (
    MotionVectorHead, DepthHead,
    compute_motion_loss, compute_depth_loss, compute_warp_loss
)
from diffsynth.models.spatiotemporal_depth_head import (
    SpatioTemporalDepthHead, SpatioTemporalDepthHeadSimple
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FinetuneAllWanTrainingModule(DiffusionTrainingModule):
    """
    Training module for fine-tuning WAN model with LoRA + motion + depth + RGB losses.

    This module loads pre-trained motion and depth heads from a checkpoint and
    continues training all components together with LoRA for the DiT backbone.
    """

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
        # Depth-specific parameters
        depth_loss_weight: float = 0.1,
        depth_loss_type: str = "mse",
        # Noise loss weight (for balancing)
        noise_loss_weight: float = 1.0,
        # Warp loss parameters
        use_warp_loss: bool = False,
        warp_loss_weight: float = 0.1,
        warp_loss_type: str = "mse",
        # Checkpoint paths for motion and depth heads
        motion_head_checkpoint: str = None,
        depth_head_checkpoint: str = None,
        # Spatio-temporal depth head parameters
        use_spatiotemporal_depth: bool = False,
        spatiotemporal_depth_type: str = "simple",
        num_temporal_heads: int = 8,
        temporal_head_dim: int = 64,
        num_temporal_blocks: int = 2,
        temporal_pos_embed_type: str = "rope",
        # Whether to freeze heads (only train LoRA)
        freeze_motion_head: bool = False,
        freeze_depth_head: bool = False,
        # Timestep sampling strategy
        timestep_sampling: str = "uniform",
        timestep_bias_strength: float = 2.0,
    ):
        super().__init__()

        # Timestep sampling
        self.timestep_sampling = timestep_sampling
        self.timestep_bias_strength = timestep_bias_strength
        if timestep_sampling not in TIMESTEP_SAMPLERS:
            print(f"[WARNING] Unknown timestep sampling '{timestep_sampling}', using 'uniform'")
            self.timestep_sampling = "uniform"

        # Load models
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths, enable_fp8_training=False
        )

        # Use local tokenizer config
        local_tokenizer_config = ModelConfig(
            path="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
        )

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=local_tokenizer_config,
        )

        # Store parameters
        self.motion_channels = motion_channels
        self.motion_loss_weight = motion_loss_weight
        self.motion_loss_type = motion_loss_type
        self.depth_loss_weight = depth_loss_weight
        self.depth_loss_type = depth_loss_type
        self.noise_loss_weight = noise_loss_weight
        self.use_warp_loss = use_warp_loss
        self.warp_loss_weight = warp_loss_weight
        self.warp_loss_type = warp_loss_type
        self.freeze_motion_head = freeze_motion_head
        self.freeze_depth_head = freeze_depth_head

        # Spatio-temporal depth parameters
        self.use_spatiotemporal_depth = use_spatiotemporal_depth
        self.spatiotemporal_depth_type = spatiotemporal_depth_type
        self.num_temporal_heads = num_temporal_heads
        self.temporal_head_dim = temporal_head_dim
        self.num_temporal_blocks = num_temporal_blocks
        self.temporal_pos_embed_type = temporal_pos_embed_type

        # Initialize feature capture variables
        self._dit_features = None
        self._dit_timestep_embed = None

        # Setup feature capture hook
        self._setup_feature_capture_hook()

        # Create and setup motion head
        self._setup_motion_head()

        # Create and setup depth head
        self._setup_depth_head()

        # Load checkpoints for motion and depth heads if provided
        if motion_head_checkpoint:
            self._load_motion_head_checkpoint(motion_head_checkpoint)
        if depth_head_checkpoint:
            self._load_depth_head_checkpoint(depth_head_checkpoint)

        # Setup LoRA for DiT backbone
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # Apply freeze settings for heads
        self._apply_head_freeze_settings()

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # Print configuration summary
        self._print_config_summary()

    def _setup_feature_capture_hook(self):
        """Setup hook to capture DiT features before the output head."""
        dit = self.pipe.dit

        def capture_features_hook(module, input, output):
            self._dit_features = input[0]
            self._dit_timestep_embed = input[1] if len(input) > 1 else None

        dit.head.register_forward_hook(capture_features_hook)

    def _setup_motion_head(self):
        """Add motion vector head to the DiT model."""
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
        self.motion_head = self.motion_head.to(dtype=dit_dtype)

    def _setup_depth_head(self):
        """Add depth prediction head to the DiT model."""
        dit = self.pipe.dit
        dim = dit.dim
        patch_size = dit.patch_size
        eps = 1e-6

        if self.use_spatiotemporal_depth:
            if self.spatiotemporal_depth_type == "full":
                self.depth_head = SpatioTemporalDepthHead(
                    dim=dim,
                    patch_size=patch_size,
                    features=256,
                    num_temporal_heads=self.num_temporal_heads,
                    temporal_head_dim=self.temporal_head_dim,
                    num_temporal_blocks=self.num_temporal_blocks,
                    use_bn=False,
                    pos_embed_type=self.temporal_pos_embed_type,
                    max_frames=256,
                    output_scale=1.0,
                    eps=eps,
                )
                print(f"[INFO] Using SpatioTemporalDepthHead (full)")
            else:
                self.depth_head = SpatioTemporalDepthHeadSimple(
                    dim=dim,
                    depth_channels=1,
                    patch_size=patch_size,
                    num_temporal_heads=self.num_temporal_heads,
                    temporal_head_dim=self.temporal_head_dim,
                    num_temporal_blocks=self.num_temporal_blocks,
                    pos_embed_type=self.temporal_pos_embed_type,
                    max_frames=256,
                    output_scale=1.0,
                    eps=eps,
                )
                print(f"[INFO] Using SpatioTemporalDepthHeadSimple")
        else:
            self.depth_head = DepthHead(
                dim=dim,
                depth_channels=1,
                patch_size=patch_size,
                eps=eps,
                output_scale=1.0,
            )
            print(f"[INFO] Using standard DepthHead")

        dit_dtype = next(dit.parameters()).dtype
        self.depth_head = self.depth_head.to(dtype=dit_dtype)

    def _load_motion_head_checkpoint(self, checkpoint_path: str):
        """Load motion head weights from checkpoint (.safetensors or .pt)."""
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Motion head checkpoint not found: {checkpoint_path}")
            return

        print(f"[INFO] Loading motion head checkpoint: {checkpoint_path}")

        # Load based on file extension
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "motion_head" in state_dict:
            motion_state = state_dict["motion_head"]
        else:
            # Assume the entire state dict is for motion head
            motion_state = state_dict

        self.motion_head.load_state_dict(motion_state, strict=False)
        print(f"[INFO] Motion head checkpoint loaded successfully")

    def _load_depth_head_checkpoint(self, checkpoint_path: str):
        """Load depth head weights from checkpoint (.safetensors or .pt)."""
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Depth head checkpoint not found: {checkpoint_path}")
            return

        print(f"[INFO] Loading depth head checkpoint: {checkpoint_path}")

        # Load based on file extension
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "depth_head" in state_dict:
            depth_state = state_dict["depth_head"]
        else:
            depth_state = state_dict

        self.depth_head.load_state_dict(depth_state, strict=False)
        print(f"[INFO] Depth head checkpoint loaded successfully")

    def _apply_head_freeze_settings(self):
        """Apply freeze settings for motion and depth heads."""
        if self.freeze_motion_head:
            for param in self.motion_head.parameters():
                param.requires_grad = False
            print("[INFO] Motion head frozen")
        else:
            for param in self.motion_head.parameters():
                param.requires_grad = True

        if self.freeze_depth_head:
            for param in self.depth_head.parameters():
                param.requires_grad = False
            print("[INFO] Depth head frozen")
        else:
            for param in self.depth_head.parameters():
                param.requires_grad = True

    def _print_config_summary(self):
        """Print training configuration summary."""
        print("\n" + "=" * 60)
        print("Fine-tune All Training Configuration")
        print("=" * 60)
        print(f"Noise loss weight: {self.noise_loss_weight}")
        print(f"Motion loss weight: {self.motion_loss_weight}")
        print(f"Depth loss weight: {self.depth_loss_weight}")
        print(f"Warp loss enabled: {self.use_warp_loss}")
        if self.use_warp_loss:
            print(f"  Weight: {self.warp_loss_weight}")
        print(f"Motion head frozen: {self.freeze_motion_head}")
        print(f"Depth head frozen: {self.freeze_depth_head}")
        print(f"Spatio-temporal depth: {self.use_spatiotemporal_depth}")

        # Timestep sampling info
        print(f"\nTimestep sampling:")
        print(f"  - Strategy: {self.timestep_sampling}")
        if self.timestep_sampling == "high_noise_bias":
            print(f"  - Bias strength: {self.timestep_bias_strength}")

        # Count trainable parameters
        motion_params = sum(p.numel() for p in self.motion_head.parameters() if p.requires_grad)
        depth_params = sum(p.numel() for p in self.depth_head.parameters() if p.requires_grad)

        print(f"\nTrainable parameters:")
        print(f"  - Motion head: {motion_params:,}")
        print(f"  - Depth head: {depth_params:,}")
        print("=" * 60 + "\n")

    def forward_preprocess(self, data):
        """Preprocess input data including motion vectors and depth maps."""
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        video_data = data.get("video", None)
        if video_data is None:
            video_data = data.get("path", data)

        if isinstance(video_data, dict):
            video_frames = video_data.get("video", None)
            motion_vectors = video_data.get("motion_vectors", None)
            depth_maps = video_data.get("depth_maps", None)
            if video_frames is None:
                raise ValueError(f"Expected 'video' key in data dict, got keys: {video_data.keys()}")
        elif isinstance(video_data, list):
            video_frames = video_data
            motion_vectors = data.get("motion_vectors", None)
            depth_maps = data.get("depth_maps", None)
        else:
            raise ValueError(f"Unexpected video_data type: {type(video_data)}")

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

        if motion_vectors is not None:
            inputs_shared["target_motion_vectors"] = motion_vectors
        if depth_maps is not None:
            inputs_shared["target_depth_maps"] = depth_maps

        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = video_frames[0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = video_frames[-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        return {**inputs_shared, **inputs_posi}

    def compute_motion_prediction(self, features: torch.Tensor, t_embed: torch.Tensor, grid_size: tuple) -> torch.Tensor:
        """Compute motion vector prediction from DiT features."""
        motion_pred = self.motion_head(features, t_embed)

        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size

        from einops import rearrange
        motion_pred = rearrange(
            motion_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
            c=self.motion_channels
        )

        return motion_pred

    def compute_depth_prediction(self, features: torch.Tensor, t_embed: torch.Tensor, grid_size: tuple) -> torch.Tensor:
        """Compute depth prediction from DiT features."""
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size

        if self.use_spatiotemporal_depth:
            if self.spatiotemporal_depth_type == "full":
                depth_pred, _ = self.depth_head(features, t_embed, grid_size)
            else:
                depth_pred, _ = self.depth_head(features, t_embed, grid_size)
                depth_pred = self.depth_head.unpatchify(depth_pred, grid_size)
        else:
            depth_pred = self.depth_head(features, t_embed)

            from einops import rearrange
            depth_pred = rearrange(
                depth_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
                f=f, h=h, w=w,
                x=patch_size[0], y=patch_size[1], z=patch_size[2],
                c=1
            )

        return depth_pred

    def forward(self, data, inputs=None, return_loss_dict=False):
        """
        Forward pass with joint noise, motion, and depth prediction loss.
        All three losses are computed and combined with their respective weights.
        """
        if inputs is None:
            inputs = self.forward_preprocess(data)

        target_motion = inputs.pop("target_motion_vectors", None)
        target_depth = inputs.pop("target_depth_maps", None)

        # Standard training loss computation
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)

        # Sample timestep using configured strategy
        if self.timestep_sampling == "high_noise_bias":
            timestep = sample_timestep_high_noise_bias(
                min_timestep_boundary, max_timestep_boundary,
                self.pipe.device, self.pipe.torch_dtype,
                bias_strength=self.timestep_bias_strength
            )
        elif self.timestep_sampling in TIMESTEP_SAMPLERS:
            timestep = TIMESTEP_SAMPLERS[self.timestep_sampling](
                min_timestep_boundary, max_timestep_boundary,
                self.pipe.device, self.pipe.torch_dtype
            )
        else:
            # Fallback to uniform
            timestep = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,)).to(
                dtype=self.pipe.torch_dtype, device=self.pipe.device
            )

        inputs["latents"] = self.pipe.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.pipe.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        # Get model prediction (this also populates self._dit_features via hook)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        noise_pred = self.pipe.model_fn(**models, **inputs, timestep=timestep)

        # Compute noise prediction loss (RGB/latent MSE loss)
        noise_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        noise_loss_weighted = self.noise_loss_weight * noise_loss * self.pipe.scheduler.training_weight(timestep)

        # Initialize loss dictionary
        loss_dict = {
            "noise_loss": noise_loss.detach().item(),
            "noise_loss_weighted": noise_loss_weighted.detach().item(),
            "timestep": timestep.item(),
        }

        # Start with noise loss
        total_loss = noise_loss_weighted

        # Get grid size for predictions
        latents = inputs["latents"]
        patch_size = self.pipe.dit.patch_size
        f = latents.shape[2] // patch_size[0]
        h = latents.shape[3] // patch_size[1]
        w = latents.shape[4] // patch_size[2]
        grid_size = (f, h, w)

        # Get time embedding
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
        t_embed = self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep)
        )

        depth_pred = None

        # Compute motion loss
        if target_motion is not None and self._dit_features is not None:
            features_for_motion = self._dit_features
            if not features_for_motion.requires_grad:
                features_for_motion = features_for_motion.detach().requires_grad_(True)

            motion_pred = self.compute_motion_prediction(features_for_motion, t_embed, grid_size)

            if torch.isnan(motion_pred).any() or torch.isinf(motion_pred).any():
                motion_pred = torch.nan_to_num(motion_pred, nan=0.0, posinf=1e6, neginf=-1e6)

            target_motion = target_motion.to(device=motion_pred.device, dtype=motion_pred.dtype)
            if torch.isnan(target_motion).any() or torch.isinf(target_motion).any():
                target_motion = torch.nan_to_num(target_motion, nan=0.0, posinf=1e6, neginf=-1e6)

            if target_motion.dim() == 4:
                target_motion = target_motion.unsqueeze(0)

            pred_f, pred_h, pred_w = motion_pred.shape[2], motion_pred.shape[3], motion_pred.shape[4]
            target_f, target_h, target_w = target_motion.shape[2], target_motion.shape[3], target_motion.shape[4]

            if (pred_f, pred_h, pred_w) != (target_f, target_h, target_w):
                target_motion = F.interpolate(
                    target_motion,
                    size=(pred_f, pred_h, pred_w),
                    mode='trilinear',
                    align_corners=False
                )

            # Normalize for stable training
            target_motion_mean = target_motion.abs().mean().clamp(min=1e-6)
            target_motion_norm = target_motion / target_motion_mean
            motion_pred_norm = motion_pred / target_motion_mean

            motion_loss = compute_motion_loss(
                motion_pred_norm, target_motion_norm,
                loss_type=self.motion_loss_type
            )

            if torch.isnan(motion_loss) or torch.isinf(motion_loss):
                motion_loss = torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype, requires_grad=True)

            loss_dict["motion_loss"] = motion_loss.detach().item()
            loss_dict["motion_loss_weighted"] = (self.motion_loss_weight * motion_loss).detach().item()

            total_loss = total_loss + self.motion_loss_weight * motion_loss

        # Compute depth loss
        if target_depth is not None and self._dit_features is not None:
            features_for_depth = self._dit_features
            if not features_for_depth.requires_grad:
                features_for_depth = features_for_depth.detach().requires_grad_(True)

            if torch.isnan(features_for_depth).any() or torch.isinf(features_for_depth).any():
                features_for_depth = torch.nan_to_num(features_for_depth, nan=0.0, posinf=1e6, neginf=-1e6)

            depth_pred = self.compute_depth_prediction(features_for_depth, t_embed, grid_size)

            if torch.isnan(depth_pred).any() or torch.isinf(depth_pred).any():
                depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=1e6, neginf=-1e6)

            target_depth = target_depth.to(device=depth_pred.device, dtype=depth_pred.dtype)
            if torch.isnan(target_depth).any() or torch.isinf(target_depth).any():
                target_depth = torch.nan_to_num(target_depth, nan=0.0, posinf=1e6, neginf=-1e6)

            if target_depth.dim() == 4:
                target_depth = target_depth.unsqueeze(0)

            pred_f, pred_h, pred_w = depth_pred.shape[2], depth_pred.shape[3], depth_pred.shape[4]
            target_f, target_h, target_w = target_depth.shape[2], target_depth.shape[3], target_depth.shape[4]

            if (pred_f, pred_h, pred_w) != (target_f, target_h, target_w):
                target_depth = F.interpolate(
                    target_depth,
                    size=(pred_f, pred_h, pred_w),
                    mode='trilinear',
                    align_corners=False
                )

            # Normalize for stable training
            target_mean = target_depth.abs().mean().clamp(min=1e-6)
            target_depth_norm = target_depth / target_mean
            depth_pred_norm = depth_pred / target_mean

            depth_loss = compute_depth_loss(
                depth_pred_norm, target_depth_norm,
                loss_type=self.depth_loss_type
            )

            if torch.isnan(depth_loss) or torch.isinf(depth_loss):
                depth_loss = torch.tensor(0.0, device=depth_pred.device, dtype=depth_pred.dtype, requires_grad=True)

            loss_dict["depth_loss"] = depth_loss.detach().item()
            loss_dict["depth_loss_weighted"] = (self.depth_loss_weight * depth_loss).detach().item()

            total_loss = total_loss + self.depth_loss_weight * depth_loss

        # Compute warp loss (temporal consistency)
        if self.use_warp_loss and target_depth is not None and target_motion is not None and depth_pred is not None:
            target_motion_for_warp = target_motion.to(device=depth_pred.device, dtype=depth_pred.dtype)
            if target_motion_for_warp.dim() == 4:
                target_motion_for_warp = target_motion_for_warp.unsqueeze(0)

            pred_f, pred_h, pred_w = depth_pred.shape[2], depth_pred.shape[3], depth_pred.shape[4]
            flow_f, flow_h, flow_w = target_motion_for_warp.shape[2], target_motion_for_warp.shape[3], target_motion_for_warp.shape[4]

            if (flow_h, flow_w) != (pred_h, pred_w):
                B_flow, C_flow = target_motion_for_warp.shape[:2]
                target_motion_for_warp = target_motion_for_warp.permute(0, 2, 1, 3, 4)
                target_motion_for_warp = target_motion_for_warp.reshape(B_flow * flow_f, C_flow, flow_h, flow_w)
                target_motion_for_warp = F.interpolate(
                    target_motion_for_warp,
                    size=(pred_h, pred_w),
                    mode='bilinear',
                    align_corners=False
                )
                target_motion_for_warp[:, 0] *= pred_w / flow_w
                target_motion_for_warp[:, 1] *= pred_h / flow_h
                target_motion_for_warp = target_motion_for_warp.reshape(B_flow, flow_f, C_flow, pred_h, pred_w)
                target_motion_for_warp = target_motion_for_warp.permute(0, 2, 1, 3, 4)

            target_depth_for_warp = target_depth.to(device=depth_pred.device, dtype=depth_pred.dtype)
            if target_depth_for_warp.dim() == 4:
                target_depth_for_warp = target_depth_for_warp.unsqueeze(0)

            target_depth_f = target_depth_for_warp.shape[2]
            if (target_depth_f, target_depth_for_warp.shape[3], target_depth_for_warp.shape[4]) != (pred_f, pred_h, pred_w):
                target_depth_for_warp = F.interpolate(
                    target_depth_for_warp,
                    size=(pred_f, pred_h, pred_w),
                    mode='trilinear',
                    align_corners=False
                )

            warp_depth_mean = target_depth_for_warp.abs().mean().clamp(min=1e-6)
            depth_pred_norm_for_warp = depth_pred / warp_depth_mean
            target_depth_norm_for_warp = target_depth_for_warp / warp_depth_mean

            warp_loss = compute_warp_loss(
                depth_pred_norm_for_warp,
                target_depth_norm_for_warp,
                target_motion_for_warp,
                loss_type=self.warp_loss_type
            )

            if torch.isnan(warp_loss) or torch.isinf(warp_loss):
                warp_loss = torch.tensor(0.0, device=depth_pred.device, dtype=depth_pred.dtype, requires_grad=True)

            loss_dict["warp_loss"] = warp_loss.detach().item()
            loss_dict["warp_loss_weighted"] = (self.warp_loss_weight * warp_loss).detach().item()

            total_loss = total_loss + self.warp_loss_weight * warp_loss

        loss_dict["total_loss"] = total_loss.detach().item()

        if return_loss_dict:
            return total_loss, loss_dict
        return total_loss

    def get_trainable_parameters(self):
        """Get all trainable parameters including motion and depth heads."""
        params = list(super().get_trainable_parameters())
        if hasattr(self, 'motion_head') and not self.freeze_motion_head:
            params.extend(self.motion_head.parameters())
        if hasattr(self, 'depth_head') and not self.freeze_depth_head:
            params.extend(self.depth_head.parameters())
        return params

    def trainable_modules(self):
        """Override to include motion and depth head parameters in optimizer."""
        try:
            base_params = list(super().trainable_modules())
        except (TypeError, AttributeError):
            base_params = []

        if hasattr(self, 'motion_head') and not self.freeze_motion_head:
            for param in self.motion_head.parameters():
                if param.requires_grad:
                    base_params.append(param)

        if hasattr(self, 'depth_head') and not self.freeze_depth_head:
            for param in self.depth_head.parameters():
                if param.requires_grad:
                    base_params.append(param)

        return base_params

    def save_auxiliary_heads(self, output_path: str, step: int = None):
        """Save motion and depth heads to separate safetensors checkpoint files."""
        os.makedirs(output_path, exist_ok=True)
        suffix = f"_step-{step}" if step else ""

        # Save motion head as safetensors
        motion_path = os.path.join(output_path, f"motion_head{suffix}.safetensors")
        # Convert state dict values to contiguous tensors for safetensors
        motion_state = {k: v.contiguous() for k, v in self.motion_head.state_dict().items()}
        save_file(motion_state, motion_path)
        print(f"[INFO] Saved motion head to: {motion_path}")

        # Save depth head as safetensors
        depth_path = os.path.join(output_path, f"depth_head{suffix}.safetensors")
        depth_state = {k: v.contiguous() for k, v in self.depth_head.state_dict().items()}
        save_file(depth_state, depth_path)
        print(f"[INFO] Saved depth head to: {depth_path}")


class LoadImageSequenceWithMotionWrapper(DataProcessingOperator):
    """Wrapper that integrates LoadImageSequenceWithMotion with the data pipeline."""

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

    def __call__(self, data: str) -> Dict[str, Any]:
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
    """Create a data operator for loading image sequences with motion vectors and depth maps."""
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


def finetune_all_parser():
    """Create argument parser for fine-tune all script."""
    parser = wan_parser()

    # Motion parameters
    parser.add_argument("--motion_channels", type=int, default=4)
    parser.add_argument("--motion_loss_weight", type=float, default=0.1)
    parser.add_argument("--motion_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--normalize_motion", action="store_true")
    parser.add_argument("--motion_scale", type=float, default=1.0)

    # Depth parameters
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)
    parser.add_argument("--depth_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1", "log_l1"])
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--normalize_depth", action="store_true")

    # Noise loss weight
    parser.add_argument("--noise_loss_weight", type=float, default=1.0, help="Weight for noise/RGB loss")

    # Warp loss
    parser.add_argument("--use_warp_loss", action="store_true")
    parser.add_argument("--warp_loss_weight", type=float, default=0.1)
    parser.add_argument("--warp_loss_type", type=str, default="mse", choices=["mse", "l1", "smooth_l1"])

    # Checkpoint paths for pre-trained heads
    parser.add_argument("--motion_head_checkpoint", type=str, default=None,
                        help="Path to pre-trained motion head checkpoint")
    parser.add_argument("--depth_head_checkpoint", type=str, default=None,
                        help="Path to pre-trained depth head checkpoint")

    # Head freeze options
    parser.add_argument("--freeze_motion_head", action="store_true",
                        help="Freeze motion head (only train LoRA)")
    parser.add_argument("--freeze_depth_head", action="store_true",
                        help="Freeze depth head (only train LoRA)")

    # Spatio-temporal depth head
    parser.add_argument("--use_spatiotemporal_depth", action="store_true")
    parser.add_argument("--spatiotemporal_depth_type", type=str, default="simple", choices=["simple", "full"])
    parser.add_argument("--num_temporal_heads", type=int, default=8)
    parser.add_argument("--temporal_head_dim", type=int, default=64)
    parser.add_argument("--num_temporal_blocks", type=int, default=2)
    parser.add_argument("--temporal_pos_embed_type", type=str, default="rope", choices=["rope", "ape"])

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="wan-finetune-all")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    # Auxiliary head saving
    parser.add_argument("--save_heads_every_n_steps", type=int, default=100,
                        help="Save motion and depth heads every N steps")

    # Timestep sampling strategy
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "high_noise_bias", "logit_normal", "logit_normal_high",
                 "cubic_high", "linear_high", "beta_high", "truncated_high"],
        help="""Timestep sampling strategy for training:
            uniform - standard uniform sampling (baseline)
            high_noise_bias - power transformation biased towards high noise (recommended)
            logit_normal - logit-normal distribution (SD3/Flux style)
            logit_normal_high - logit-normal biased towards high noise
            cubic_high - cubic transformation for high noise bias
            linear_high - linear increasing probability towards high noise
            beta_high - beta distribution biased towards high noise
            truncated_high - 70% chance to sample from top 50% timesteps
        """
    )
    parser.add_argument(
        "--timestep_bias_strength",
        type=float,
        default=2.0,
        help="Bias strength for high_noise_bias sampling (higher = more high noise samples)"
    )

    return parser


def launch_finetune_all_training(
    dataset: torch.utils.data.Dataset,
    model: FinetuneAllWanTrainingModule,
    model_logger: ModelLogger,
    args,
):
    """Launch training with all losses and wandb logging."""
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters

    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "noise_loss_weight": args.noise_loss_weight,
            "motion_loss_weight": args.motion_loss_weight,
            "motion_loss_type": args.motion_loss_type,
            "depth_loss_weight": args.depth_loss_weight,
            "depth_loss_type": args.depth_loss_type,
            "use_warp_loss": args.use_warp_loss,
            "warp_loss_weight": args.warp_loss_weight,
            "motion_channels": args.motion_channels,
            "lora_rank": args.lora_rank if hasattr(args, 'lora_rank') else None,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "freeze_motion_head": args.freeze_motion_head,
            "freeze_depth_head": args.freeze_depth_head,
            "timestep_sampling": args.timestep_sampling,
            "timestep_bias_strength": args.timestep_bias_strength,
        }

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    log_every_n_steps = args.log_every_n_steps
    save_heads_every_n_steps = args.save_heads_every_n_steps

    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wandb_config,
        )
        print(f"=" * 50)
        print(f"Wandb initialized: {wandb.run.name}")
        print(f"Run URL: {wandb.run.get_url()}")
        print(f"=" * 50)

    for epoch_id in range(num_epochs):
        epoch_losses = {
            "noise_loss": [],
            "motion_loss": [],
            "depth_loss": [],
            "warp_loss": [],
            "total_loss": [],
        }

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id + 1}/{num_epochs}", disable=not accelerator.is_main_process)

        for data in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                unwrapped_model = accelerator.unwrap_model(model)

                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    result = unwrapped_model({}, inputs=data, return_loss_dict=True)
                else:
                    result = unwrapped_model(data, return_loss_dict=True)

                if isinstance(result, tuple) and len(result) == 2:
                    loss, loss_dict = result
                else:
                    loss = result
                    loss_dict = {"total_loss": loss.detach().item()}

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                global_step += 1

                # Collect losses
                epoch_losses["noise_loss"].append(loss_dict.get("noise_loss", 0))
                epoch_losses["total_loss"].append(loss_dict.get("total_loss", 0))
                if "motion_loss" in loss_dict:
                    epoch_losses["motion_loss"].append(loss_dict["motion_loss"])
                if "depth_loss" in loss_dict:
                    epoch_losses["depth_loss"].append(loss_dict["depth_loss"])
                if "warp_loss" in loss_dict:
                    epoch_losses["warp_loss"].append(loss_dict["warp_loss"])

                # Update progress bar
                postfix_dict = {
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "noise": f"{loss_dict.get('noise_loss', 0):.4f}",
                    "motion": f"{loss_dict.get('motion_loss', 0):.4f}",
                    "depth": f"{loss_dict.get('depth_loss', 0):.4f}",
                }
                if "warp_loss" in loss_dict:
                    postfix_dict["warp"] = f"{loss_dict['warp_loss']:.4f}"
                progress_bar.set_postfix(postfix_dict)

                # Log to wandb
                if use_wandb and accelerator.is_main_process and global_step % log_every_n_steps == 0:
                    log_data = {
                        "train/total_loss": loss_dict["total_loss"],
                        "train/noise_loss": loss_dict.get("noise_loss", 0),
                        "train/noise_loss_weighted": loss_dict.get("noise_loss_weighted", 0),
                        "train/timestep": loss_dict.get("timestep", 0),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                        "train/global_step": global_step,
                    }

                    if "motion_loss" in loss_dict:
                        log_data["train/motion_loss"] = loss_dict["motion_loss"]
                        log_data["train/motion_loss_weighted"] = loss_dict.get("motion_loss_weighted", 0)

                    if "depth_loss" in loss_dict:
                        log_data["train/depth_loss"] = loss_dict["depth_loss"]
                        log_data["train/depth_loss_weighted"] = loss_dict.get("depth_loss_weighted", 0)

                    if "warp_loss" in loss_dict:
                        log_data["train/warp_loss"] = loss_dict["warp_loss"]
                        log_data["train/warp_loss_weighted"] = loss_dict.get("warp_loss_weighted", 0)

                    wandb.log(log_data, step=global_step)

                # Save checkpoints
                model_logger.on_step_end(accelerator, model, save_steps)

                # Save auxiliary heads
                if accelerator.is_main_process and save_heads_every_n_steps and global_step % save_heads_every_n_steps == 0:
                    unwrapped_model.save_auxiliary_heads(args.output_path, step=global_step)

        # Log epoch summary
        if use_wandb and accelerator.is_main_process:
            epoch_summary = {
                "epoch/noise_loss_avg": sum(epoch_losses["noise_loss"]) / len(epoch_losses["noise_loss"]) if epoch_losses["noise_loss"] else 0,
                "epoch/total_loss_avg": sum(epoch_losses["total_loss"]) / len(epoch_losses["total_loss"]) if epoch_losses["total_loss"] else 0,
                "epoch/epoch": epoch_id,
            }

            if epoch_losses["motion_loss"]:
                epoch_summary["epoch/motion_loss_avg"] = sum(epoch_losses["motion_loss"]) / len(epoch_losses["motion_loss"])
            if epoch_losses["depth_loss"]:
                epoch_summary["epoch/depth_loss_avg"] = sum(epoch_losses["depth_loss"]) / len(epoch_losses["depth_loss"])
            if epoch_losses["warp_loss"]:
                epoch_summary["epoch/warp_loss_avg"] = sum(epoch_losses["warp_loss"]) / len(epoch_losses["warp_loss"])

            wandb.log(epoch_summary, step=global_step)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    # Final save
    model_logger.on_training_end(accelerator, model, save_steps)

    # Save final auxiliary heads
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_auxiliary_heads(args.output_path, step=global_step)

    if use_wandb and accelerator.is_main_process:
        wandb.finish()
        print("Wandb run finished.")


if __name__ == "__main__":
    parser = finetune_all_parser()
    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("WAN Fine-tune All (LoRA + Motion + Depth + RGB)")
    print("=" * 60)
    print(f"Noise loss weight: {args.noise_loss_weight}")
    print(f"Motion loss weight: {args.motion_loss_weight}")
    print(f"Depth loss weight: {args.depth_loss_weight}")
    print(f"Warp loss enabled: {args.use_warp_loss}")
    if args.use_warp_loss:
        print(f"  Weight: {args.warp_loss_weight}")
    print(f"Motion head checkpoint: {args.motion_head_checkpoint or 'None (random init)'}")
    print(f"Depth head checkpoint: {args.depth_head_checkpoint or 'None (random init)'}")
    print(f"Freeze motion head: {args.freeze_motion_head}")
    print(f"Freeze depth head: {args.freeze_depth_head}")
    print(f"Spatio-temporal depth: {args.use_spatiotemporal_depth}")
    if args.use_spatiotemporal_depth:
        print(f"  Type: {args.spatiotemporal_depth_type}")
    print(f"Timestep sampling: {args.timestep_sampling}")
    if args.timestep_sampling == "high_noise_bias":
        print(f"  Bias strength: {args.timestep_bias_strength}")
    print(f"Wandb enabled: {args.use_wandb}")
    print("=" * 60)

    # Create dataset
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=create_motion_data_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            motion_channels=args.motion_channels,
            normalize_motion=args.normalize_motion,
            motion_scale=args.motion_scale,
            depth_scale=args.depth_scale,
            normalize_depth=args.normalize_depth,
            load_depth=True,
        ),
    )

    # Create model
    model = FinetuneAllWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        motion_channels=args.motion_channels,
        motion_loss_weight=args.motion_loss_weight,
        motion_loss_type=args.motion_loss_type,
        depth_loss_weight=args.depth_loss_weight,
        depth_loss_type=args.depth_loss_type,
        noise_loss_weight=args.noise_loss_weight,
        use_warp_loss=args.use_warp_loss,
        warp_loss_weight=args.warp_loss_weight,
        warp_loss_type=args.warp_loss_type,
        motion_head_checkpoint=args.motion_head_checkpoint,
        depth_head_checkpoint=args.depth_head_checkpoint,
        use_spatiotemporal_depth=args.use_spatiotemporal_depth,
        spatiotemporal_depth_type=args.spatiotemporal_depth_type,
        num_temporal_heads=args.num_temporal_heads,
        temporal_head_dim=args.temporal_head_dim,
        num_temporal_blocks=args.num_temporal_blocks,
        temporal_pos_embed_type=args.temporal_pos_embed_type,
        freeze_motion_head=args.freeze_motion_head,
        freeze_depth_head=args.freeze_depth_head,
        timestep_sampling=args.timestep_sampling,
        timestep_bias_strength=args.timestep_bias_strength,
    )

    # Create model logger
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)

    # Launch training
    launch_finetune_all_training(dataset, model, model_logger, args=args)
