"""
Training script for WAN2.2-5B Text-to-Image-to-Video with Motion and Depth Heads.

Two-stage training:
- Stage 1: Train motion head + depth head only (backbone frozen)
- Stage 2: Train backbone (LoRA) + motion head + depth head together
"""

import torch
import os
import json
import argparse
from tqdm import tqdm
import wandb

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule
from diffsynth.trainers.unified_dataset import (
    UnifiedDataset,
    DataProcessingOperator,
)
from diffsynth.trainers.image_sequence_loader import LoadImageSequenceWithMotion
from diffsynth.models.wan_video_dit_motion import (
    MotionVectorHead, DepthHead,
    compute_motion_loss, compute_depth_loss, compute_warp_loss
)
from diffsynth.models.rgb_warp_loss import compute_rgb_warp_loss
from diffsynth.models.spatiotemporal_depth_head import (
    SpatioTemporalDepthHead, SpatioTemporalDepthHeadSimple
)

from accelerate import Accelerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        motion_channels: int = 2,
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
    ):
        super().__init__()

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
        self._dit_timestep_embed = None

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
        """Setup hook to capture DiT features for motion/depth heads."""
        def hook_fn(module, input, output):
            self._dit_features = output
            return output

        # Register hook on the last DiT block
        if hasattr(self.pipe.dit, 'blocks'):
            self.pipe.dit.blocks[-1].register_forward_hook(hook_fn)
        elif hasattr(self.pipe.dit, 'transformer_blocks'):
            self.pipe.dit.transformer_blocks[-1].register_forward_hook(hook_fn)

    def _setup_motion_head(self, checkpoint_path=None):
        """Create and attach motion vector head."""
        # Get DiT dimension
        dit_dim = self.pipe.dit.hidden_size if hasattr(self.pipe.dit, 'hidden_size') else 3072

        # Get patch size
        if hasattr(self.pipe.dit, 'patch_size'):
            patch_size = self.pipe.dit.patch_size
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size, patch_size)
        else:
            patch_size = (1, 2, 2)  # Default for WAN models

        self.motion_head = MotionVectorHead(
            in_channels=dit_dim,
            motion_channels=self.motion_channels,
            patch_size=patch_size,
        ).to(self.pipe.device)

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading motion head checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.pipe.device)
            self.motion_head.load_state_dict(state_dict)

        print(f"Motion head initialized with {self.motion_channels} channels")

    def _setup_depth_head(self, checkpoint_path=None):
        """Create and attach depth head."""
        # Get DiT dimension
        dit_dim = self.pipe.dit.hidden_size if hasattr(self.pipe.dit, 'hidden_size') else 3072

        # Get patch size
        if hasattr(self.pipe.dit, 'patch_size'):
            patch_size = self.pipe.dit.patch_size
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size, patch_size)
        else:
            patch_size = (1, 2, 2)

        if self.use_spatiotemporal_depth:
            # Use spatio-temporal depth head
            if self.spatiotemporal_depth_type == "full":
                self.depth_head = SpatioTemporalDepthHead(
                    in_channels=dit_dim,
                    patch_size=patch_size,
                    num_heads=self.num_temporal_heads,
                    head_dim=self.temporal_head_dim,
                    num_blocks=self.num_temporal_blocks,
                    pos_embed_type=self.temporal_pos_embed_type,
                ).to(self.pipe.device)
            else:
                self.depth_head = SpatioTemporalDepthHeadSimple(
                    in_channels=dit_dim,
                    patch_size=patch_size,
                    num_heads=self.num_temporal_heads,
                    head_dim=self.temporal_head_dim,
                    num_blocks=self.num_temporal_blocks,
                    pos_embed_type=self.temporal_pos_embed_type,
                ).to(self.pipe.device)
            print(f"Spatio-temporal depth head initialized ({self.spatiotemporal_depth_type})")
        else:
            # Use standard depth head
            self.depth_head = DepthHead(
                in_channels=dit_dim,
                patch_size=patch_size,
            ).to(self.pipe.device)
            print("Standard depth head initialized")

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading depth head checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.pipe.device)
            self.depth_head.load_state_dict(state_dict)

    def _setup_training_mode(
        self, trainable_models, lora_base_model,
        lora_target_modules, lora_rank, lora_checkpoint
    ):
        """Setup training mode: freeze/unfreeze layers, add LoRA, etc."""

        if self.training_mode == "heads_only":
            # Stage 1: Train only motion and depth heads, freeze everything else
            print("=" * 60)
            print("TRAINING MODE: Heads Only (Stage 1)")
            print("=" * 60)

            # Freeze all pipeline models
            for name in self.pipe.model_names:
                model = getattr(self.pipe, name)
                for param in model.parameters():
                    param.requires_grad = False

            # Unfreeze motion head
            if self.enable_motion:
                for param in self.motion_head.parameters():
                    param.requires_grad = True
                print("✓ Motion head: TRAINABLE")

            # Unfreeze depth head
            if self.enable_depth:
                for param in self.depth_head.parameters():
                    param.requires_grad = True
                print("✓ Depth head: TRAINABLE")

            print("✗ DiT backbone: FROZEN")
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

            # Ensure motion head is trainable
            if self.enable_motion:
                for param in self.motion_head.parameters():
                    param.requires_grad = True
                print("✓ Motion head: TRAINABLE")

            # Ensure depth head is trainable
            if self.enable_depth:
                for param in self.depth_head.parameters():
                    param.requires_grad = True
                print("✓ Depth head: TRAINABLE")

            print(f"✓ DiT backbone: LoRA (rank={lora_rank})")
            print("=" * 60)

        elif self.training_mode == "both_last":
            # Alternative: Train heads + last DiT layer only
            print("=" * 60)
            print("TRAINING MODE: Heads + Last DiT Layer")
            print("=" * 60)

            # Freeze all pipeline models
            for name in self.pipe.model_names:
                model = getattr(self.pipe, name)
                for param in model.parameters():
                    param.requires_grad = False

            # Unfreeze last DiT block
            if hasattr(self.pipe.dit, 'blocks'):
                for param in self.pipe.dit.blocks[-1].parameters():
                    param.requires_grad = True
            elif hasattr(self.pipe.dit, 'transformer_blocks'):
                for param in self.pipe.dit.transformer_blocks[-1].parameters():
                    param.requires_grad = True

            # Unfreeze heads
            if self.enable_motion:
                for param in self.motion_head.parameters():
                    param.requires_grad = True
                print("✓ Motion head: TRAINABLE")

            if self.enable_depth:
                for param in self.depth_head.parameters():
                    param.requires_grad = True
                print("✓ Depth head: TRAINABLE")

            print("✓ DiT last layer: TRAINABLE")
            print("=" * 60)

    def forward_preprocess(self, data):
        """Preprocess data for forward pass."""
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
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
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Store motion and depth ground truth if available
        if "motion_flows" in data:
            inputs_shared["target_motion_flows"] = data["motion_flows"]
        if "depth_maps" in data:
            inputs_shared["target_depth_maps"] = data["depth_maps"]
        if self.use_rgb_warp_loss:
            inputs_shared["target_rgb_frames"] = data["video"]

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
        target_motion = inputs.pop("target_motion_flows", None)
        target_depth = inputs.pop("target_depth_maps", None)
        target_rgb_frames = inputs.pop("target_rgb_frames", None)

        # Get models for training loss
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}

        # Compute standard diffusion loss
        noise_loss = self.pipe.training_loss(**models, **inputs)

        # Reset feature capture
        self._dit_features = None

        # Run a forward pass to capture DiT features
        with torch.no_grad():
            _ = self.pipe.training_loss(**models, **inputs)

        # Initialize loss dict
        loss_dict = {
            "noise_loss": noise_loss.detach().item(),
            "total_loss": noise_loss.detach().item(),
        }
        total_loss = noise_loss

        # Compute motion loss
        if self.enable_motion and self._dit_features is not None and target_motion is not None:
            motion_pred = self.motion_head(self._dit_features)

            # Align shapes
            target_motion = target_motion.to(device=motion_pred.device, dtype=motion_pred.dtype)
            if target_motion.shape != motion_pred.shape:
                import torch.nn.functional as F
                target_motion = F.interpolate(
                    target_motion,
                    size=motion_pred.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            motion_loss = compute_motion_loss(
                motion_pred, target_motion, loss_type=self.motion_loss_type
            )

            loss_dict["motion_loss"] = motion_loss.detach().item()
            total_loss = total_loss + self.motion_loss_weight * motion_loss

        # Compute depth loss
        if self.enable_depth and self._dit_features is not None and target_depth is not None:
            depth_pred = self.depth_head(self._dit_features)

            # Align shapes
            target_depth = target_depth.to(device=depth_pred.device, dtype=depth_pred.dtype)
            if target_depth.shape != depth_pred.shape:
                import torch.nn.functional as F
                target_depth = F.interpolate(
                    target_depth,
                    size=depth_pred.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            depth_loss = compute_depth_loss(
                depth_pred, target_depth, loss_type=self.depth_loss_type
            )

            loss_dict["depth_loss"] = depth_loss.detach().item()
            total_loss = total_loss + self.depth_loss_weight * depth_loss

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
    parser.add_argument("--motion_loss_weight", type=float, default=0.1)
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)

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

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    # Initialize wandb
    if args.use_wandb and accelerator.is_main_process:
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
            motion_channels=4,
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
    )

    # Create model
    model = WAN22MotionDepthTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        training_mode=args.training_mode,
        trainable_models="dit" if args.training_mode == "lora" else None,
        lora_base_model="dit" if args.training_mode == "lora" else None,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        extra_inputs="input_image",
        motion_loss_weight=args.motion_loss_weight,
        depth_loss_weight=args.depth_loss_weight,
        motion_head_checkpoint=args.motion_head_checkpoint,
        depth_head_checkpoint=args.depth_head_checkpoint,
        use_spatiotemporal_depth=args.use_spatiotemporal_depth,
        spatiotemporal_depth_type=args.spatiotemporal_depth_type,
    )

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
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

                # Save motion head
                if model.module.enable_motion:
                    torch.save(
                        model.module.motion_head.state_dict(),
                        os.path.join(save_path, "motion_head.pth")
                    )

                # Save depth head
                if model.module.enable_depth:
                    torch.save(
                        model.module.depth_head.state_dict(),
                        os.path.join(save_path, "depth_head.pth")
                    )

                # Save LoRA if in stage 2
                if args.training_mode == "lora":
                    # Save LoRA weights
                    lora_state_dict = {}
                    for name, param in model.module.pipe.dit.named_parameters():
                        if "lora" in name and param.requires_grad:
                            lora_state_dict[name] = param.cpu()
                    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pth"))

                print(f"Checkpoint saved to {save_path}")

    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_path, "final")
        os.makedirs(final_path, exist_ok=True)

        if model.module.enable_motion:
            torch.save(
                model.module.motion_head.state_dict(),
                os.path.join(final_path, "motion_head.pth")
            )

        if model.module.enable_depth:
            torch.save(
                model.module.depth_head.state_dict(),
                os.path.join(final_path, "depth_head.pth")
            )

        if args.training_mode == "lora":
            lora_state_dict = {}
            for name, param in model.module.pipe.dit.named_parameters():
                if "lora" in name and param.requires_grad:
                    lora_state_dict[name] = param.cpu()
            torch.save(lora_state_dict, os.path.join(final_path, "lora_weights.pth"))

        print(f"Final checkpoint saved to {final_path}")

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
