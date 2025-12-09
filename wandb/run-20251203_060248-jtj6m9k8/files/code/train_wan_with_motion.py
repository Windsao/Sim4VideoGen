#!/usr/bin/env python3
"""
Fine-tuning script for WAN model with motion vector and depth prediction.

This script extends the training to jointly optimize for:
1. Standard video generation (noise prediction loss)
2. Motion vector prediction (motion loss)
3. Depth prediction (depth loss)

The model learns to predict the denoised video, motion vectors between frames,
and depth maps, enabling physics-aware video generation with 3D understanding.

Usage:
    accelerate launch train_wan_with_motion.py \
        --dataset_base_path /path/to/dataset \
        --dataset_metadata_path data/metadata.csv \
        --height 480 \
        --width 832 \
        --num_frames 81 \
        --model_paths '["/path/to/model1.safetensors", ...]' \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --output_path ./output/model_lora \
        --lora_base_model "dit" \
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
        --lora_rank 32 \
        --motion_loss_weight 0.1 \
        --motion_channels 4 \
        --depth_loss_weight 0.1
"""

import torch
import torch.nn.functional as F
import os
import argparse
from typing import Optional, Dict, Any
from tqdm import tqdm

from diffsynth import load_state_dict
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
    compute_motion_loss, compute_depth_loss
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


class MotionAwareWanTrainingModule(DiffusionTrainingModule):
    """
    Training module for WAN model with motion vector and depth prediction.

    This module adds motion vector and depth heads to the DiT model and trains
    them jointly with the standard denoising objective for physics-aware
    video generation.
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
        # Training mode: controls which components to train
        # "lora" - LoRA + motion head + depth head (full adaptation)
        # "motion_last" - motion head + DiT last layer only
        # "depth_last" - depth head + DiT last layer only
        # "both_last" - motion head + depth head + DiT last layer
        # "motion_only" - motion head only (no video improvement)
        # "depth_only" - depth head only (no video improvement)
        training_mode: str = "lora",
        # Depth-specific parameters
        depth_loss_weight: float = 0.1,
        depth_loss_type: str = "mse",
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

        # Determine which heads to enable based on training mode
        self.enable_motion = training_mode in ["lora", "motion_last", "both_last", "motion_only"]
        self.enable_depth = training_mode in ["lora", "depth_last", "both_last", "depth_only"]

        # Motion parameters
        self.motion_channels = motion_channels
        self.motion_loss_weight = motion_loss_weight if self.enable_motion else 0.0
        self.motion_loss_type = motion_loss_type

        # Depth parameters
        self.depth_loss_weight = depth_loss_weight if self.enable_depth else 0.0
        self.depth_loss_type = depth_loss_type

        # Initialize feature capture variables
        self._dit_features = None
        self._dit_timestep_embed = None

        # Setup feature capture hook if either motion or depth is enabled
        if self.enable_motion or self.enable_depth:
            self._setup_feature_capture_hook()

        # Create and attach motion head if enabled
        if self.enable_motion:
            self._setup_motion_head()

        # Create and attach depth head if enabled
        if self.enable_depth:
            self._setup_depth_head()

        # Training mode setup
        if training_mode in ["motion_last", "depth_last", "both_last", "motion_only", "depth_only"]:
            # For non-LoRA modes: do essential setup without LoRA
            # Set scheduler to training mode (this is critical!)
            self.pipe.scheduler.set_timesteps(1000, training=True)
            # Freeze all models first, we'll unfreeze specific components later
            self.pipe.freeze_except([])
            # Apply appropriate freeze pattern based on mode
            self._apply_training_mode_freeze()
        else:
            # LoRA mode: full setup with LoRA
            self.switch_pipe_to_training_mode(
                self.pipe,
                trainable_models,
                lora_base_model,
                lora_target_modules,
                lora_rank,
                lora_checkpoint=lora_checkpoint,
                enable_fp8_training=False,
            )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _setup_feature_capture_hook(self):
        """Setup hook to capture DiT features before the output head."""
        dit = self.pipe.dit

        # Hook to capture DiT features before the output head
        def capture_features_hook(module, input, output):
            # The input to the head is the features we want
            # Don't use clone() as it can break gradient flow in some cases
            # Instead, store the tensor directly
            self._dit_features = input[0]
            self._dit_timestep_embed = input[1] if len(input) > 1 else None

        # Register hook on the original head
        dit.head.register_forward_hook(capture_features_hook)

    def _setup_motion_head(self):
        """Add motion vector head to the DiT model."""
        dit = self.pipe.dit

        # Get model dimensions
        dim = dit.dim
        patch_size = dit.patch_size
        eps = 1e-6

        # Create motion head
        self.motion_head = MotionVectorHead(
            dim=dim,
            motion_channels=self.motion_channels,
            patch_size=patch_size,
            eps=eps,
            output_scale=1.0,
        )

        # Convert motion head to same dtype as DiT model (bfloat16)
        # Get the dtype from the DiT's parameters
        dit_dtype = next(dit.parameters()).dtype
        self.motion_head = self.motion_head.to(dtype=dit_dtype)

    def _setup_depth_head(self):
        """Add depth prediction head to the DiT model."""
        dit = self.pipe.dit

        # Get model dimensions
        dim = dit.dim
        patch_size = dit.patch_size
        eps = 1e-6

        # Create depth head (single channel output)
        self.depth_head = DepthHead(
            dim=dim,
            depth_channels=1,
            patch_size=patch_size,
            eps=eps,
            output_scale=1.0,
        )

        # Convert depth head to same dtype as DiT model (bfloat16)
        dit_dtype = next(dit.parameters()).dtype
        self.depth_head = self.depth_head.to(dtype=dit_dtype)

    def _apply_training_mode_freeze(self):
        """Apply freeze pattern based on training mode."""
        dit = self.pipe.dit
        motion_params = 0
        depth_params = 0
        dit_head_params = 0

        # Determine what to unfreeze based on training mode
        train_motion = self.training_mode in ["motion_last", "both_last", "motion_only"]
        train_depth = self.training_mode in ["depth_last", "both_last", "depth_only"]
        train_dit_head = self.training_mode in ["motion_last", "depth_last", "both_last"]

        # Unfreeze motion head if applicable
        if train_motion and hasattr(self, 'motion_head'):
            for param in self.motion_head.parameters():
                param.requires_grad = True
            motion_params = sum(p.numel() for p in self.motion_head.parameters() if p.requires_grad)

        # Unfreeze depth head if applicable
        if train_depth and hasattr(self, 'depth_head'):
            for param in self.depth_head.parameters():
                param.requires_grad = True
            depth_params = sum(p.numel() for p in self.depth_head.parameters() if p.requires_grad)

        # Unfreeze DiT output head if applicable
        if train_dit_head:
            for param in dit.head.parameters():
                param.requires_grad = True
            dit_head_params = sum(p.numel() for p in dit.head.parameters() if p.requires_grad)

        # Print training configuration
        mode_descriptions = {
            "motion_last": "Motion Head + DiT Last Layer",
            "depth_last": "Depth Head + DiT Last Layer",
            "both_last": "Motion Head + Depth Head + DiT Last Layer",
            "motion_only": "Motion Head Only",
            "depth_only": "Depth Head Only",
        }
        print(f"\nTraining mode: {mode_descriptions.get(self.training_mode, self.training_mode)}")
        print(f"Trainable parameters:")
        if motion_params > 0:
            print(f"  - Motion head: {motion_params:,}")
        if depth_params > 0:
            print(f"  - Depth head: {depth_params:,}")
        if dit_head_params > 0:
            print(f"  - DiT output head: {dit_head_params:,}")
        print(f"  - Total: {motion_params + depth_params + dit_head_params:,}")

        # Verify heads are properly registered as submodules
        print(f"\nRegistered submodules:")
        for name, module in self.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  - {name}: {param_count:,} params ({trainable:,} trainable)")
        print()

    def forward_preprocess(self, data):
        """Preprocess input data including motion vectors."""
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # Handle video data - it might be a dict from LoadImageSequenceWithMotion
        # The data operator processes "video" key which may return a dict with 'video' and 'motion_vectors'
        video_data = data.get("video", None)
        if video_data is None:
            # Fallback to "path" key if "video" is not present
            video_data = data.get("path", data)

        if isinstance(video_data, dict):
            video_frames = video_data.get("video", None)
            motion_vectors = video_data.get("motion_vectors", None)
            depth_maps = video_data.get("depth_maps", None)
            # If video_frames is still None, the dict structure is unexpected
            if video_frames is None:
                raise ValueError(f"Expected 'video' key in data dict, got keys: {video_data.keys()}")
        elif isinstance(video_data, list):
            # Direct list of frames
            video_frames = video_data
            motion_vectors = data.get("motion_vectors", None)
            depth_maps = data.get("depth_maps", None)
        else:
            raise ValueError(f"Unexpected video_data type: {type(video_data)}. Expected dict or list.")

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

        # Store motion vectors for loss computation
        if motion_vectors is not None:
            inputs_shared["target_motion_vectors"] = motion_vectors

        # Store depth maps for loss computation
        if depth_maps is not None:
            inputs_shared["target_depth_maps"] = depth_maps

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = video_frames[0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = video_frames[-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        return {**inputs_shared, **inputs_posi}

    def compute_motion_prediction(self, features: torch.Tensor, t_embed: torch.Tensor, grid_size: tuple) -> torch.Tensor:
        """
        Compute motion vector prediction from DiT features.

        Args:
            features: DiT features of shape (B, S, D)
            t_embed: Time embedding of shape (B, D)
            grid_size: (F, H, W) grid dimensions

        Returns:
            Motion vectors of shape (B, C, F, H, W)
        """
        motion_pred = self.motion_head(features, t_embed)

        # Unpatchify
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
        """
        Compute depth prediction from DiT features.

        Args:
            features: DiT features of shape (B, S, D)
            t_embed: Time embedding of shape (B, D)
            grid_size: (F, H, W) grid dimensions

        Returns:
            Depth maps of shape (B, 1, F, H, W)
        """
        depth_pred = self.depth_head(features, t_embed)

        # Unpatchify
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size

        from einops import rearrange
        depth_pred = rearrange(
            depth_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
            c=1  # Single channel depth
        )

        return depth_pred

    def forward(self, data, inputs=None, return_loss_dict=False):
        """
        Forward pass with joint noise, motion, and depth prediction loss.

        Args:
            data: Input data batch
            inputs: Pre-processed inputs (optional)
            return_loss_dict: If True, return dict of individual losses for logging

        Returns:
            If return_loss_dict is False: total_loss (scalar)
            If return_loss_dict is True: (total_loss, loss_dict)
        """
        if inputs is None:
            inputs = self.forward_preprocess(data)

        # Get target motion vectors and depth maps if available
        target_motion = inputs.pop("target_motion_vectors", None)
        target_depth = inputs.pop("target_depth_maps", None)

        # Standard training loss computation
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.pipe.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.pipe.scheduler.num_train_timesteps)
        # Generate timestep directly as a value, not as an index into timesteps array
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
        noise_loss_weighted = noise_loss * self.pipe.scheduler.training_weight(timestep)

        # Initialize loss dictionary for logging
        loss_dict = {
            "noise_loss": noise_loss.detach().item(),
            "noise_loss_weighted": noise_loss_weighted.detach().item(),
            "timestep": timestep.item(),
        }

        # For motion_only and depth_only modes, we don't include noise_loss in total_loss
        # because the DiT is frozen and noise_loss has no gradients
        include_noise_loss = self.training_mode not in ["motion_only", "depth_only"]

        if include_noise_loss:
            total_loss = noise_loss_weighted
        else:
            # Initialize with a zero tensor that has requires_grad=True
            # This will be replaced when we add motion/depth loss
            total_loss = None

        motion_loss = None
        depth_loss = None

        # Compute motion loss if motion is enabled and target vectors are available
        if self.enable_motion and target_motion is not None and self._dit_features is not None:
            # Get grid size from latents
            latents = inputs["latents"]
            patch_size = self.pipe.dit.patch_size
            f = latents.shape[2] // patch_size[0]
            h = latents.shape[3] // patch_size[1]
            w = latents.shape[4] // patch_size[2]
            grid_size = (f, h, w)

            # Get time embedding for motion head
            from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
            t_embed = self.pipe.dit.time_embedding(
                sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep)
            )

            # Compute motion prediction
            # Ensure features have gradient tracking for auxiliary heads
            features_for_motion = self._dit_features
            if not features_for_motion.requires_grad:
                # Enable gradient tracking - the motion head params will create the grad fn
                features_for_motion = features_for_motion.detach().requires_grad_(True)

            motion_pred = self.compute_motion_prediction(
                features_for_motion, t_embed, grid_size
            )

            # Check and clamp motion prediction to prevent NaN
            if torch.isnan(motion_pred).any() or torch.isinf(motion_pred).any():
                print(f"[WARNING] NaN/Inf in motion_pred, clamping...")
                motion_pred = torch.nan_to_num(motion_pred, nan=0.0, posinf=1e6, neginf=-1e6)

            # Resize target motion to match prediction size if needed
            target_motion = target_motion.to(device=motion_pred.device, dtype=motion_pred.dtype)

            # Check and clamp target motion
            if torch.isnan(target_motion).any() or torch.isinf(target_motion).any():
                print(f"[WARNING] NaN/Inf in target_motion, clamping...")
                target_motion = torch.nan_to_num(target_motion, nan=0.0, posinf=1e6, neginf=-1e6)

            # Motion vectors have shape (C, F-1, H, W) but prediction has (B, C, F, H, W)
            # We need to handle the batch dimension and potential size mismatches
            if target_motion.dim() == 4:
                target_motion = target_motion.unsqueeze(0)  # Add batch dim

            # Resize target motion to match prediction dimensions (temporal and spatial)
            # The prediction is in latent space which has different dimensions than pixel space
            pred_f, pred_h, pred_w = motion_pred.shape[2], motion_pred.shape[3], motion_pred.shape[4]
            target_f, target_h, target_w = target_motion.shape[2], target_motion.shape[3], target_motion.shape[4]

            if (pred_f, pred_h, pred_w) != (target_f, target_h, target_w):
                # Use 3D interpolation to resize target to match prediction
                # target_motion shape: (B, C, F, H, W)
                target_motion = F.interpolate(
                    target_motion,
                    size=(pred_f, pred_h, pred_w),
                    mode='trilinear',
                    align_corners=False
                )

            # Normalize both predictions and targets to similar scales for stable training
            target_motion_mean = target_motion.abs().mean().clamp(min=1e-6)
            target_motion_norm = target_motion / target_motion_mean
            motion_pred_norm = motion_pred / target_motion_mean

            # Compute motion loss on normalized values
            motion_loss = compute_motion_loss(
                motion_pred_norm, target_motion_norm,
                loss_type=self.motion_loss_type
            )

            # Final NaN check on motion loss
            if torch.isnan(motion_loss) or torch.isinf(motion_loss):
                print(f"[WARNING] NaN/Inf motion_loss detected!")
                print(f"  motion_pred range: [{motion_pred.min().item():.4f}, {motion_pred.max().item():.4f}]")
                print(f"  target_motion range: [{target_motion.min().item():.4f}, {target_motion.max().item():.4f}]")
                motion_loss = torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype, requires_grad=True)

            loss_dict["motion_loss"] = motion_loss.detach().item()
            loss_dict["motion_loss_weighted"] = (self.motion_loss_weight * motion_loss).detach().item()

            weighted_motion_loss = self.motion_loss_weight * motion_loss
            if total_loss is None:
                total_loss = weighted_motion_loss
            else:
                total_loss = total_loss + weighted_motion_loss

        # Compute depth loss if depth maps are available and depth is enabled
        if self.enable_depth and target_depth is not None and self._dit_features is not None:
            # Get grid size from latents (same as motion)
            latents = inputs["latents"]
            patch_size = self.pipe.dit.patch_size
            f = latents.shape[2] // patch_size[0]
            h = latents.shape[3] // patch_size[1]
            w = latents.shape[4] // patch_size[2]
            grid_size = (f, h, w)

            # Get time embedding for depth head
            from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
            t_embed = self.pipe.dit.time_embedding(
                sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep)
            )

            # Compute depth prediction
            # Ensure features have gradient tracking for auxiliary heads
            features_for_depth = self._dit_features
            if not features_for_depth.requires_grad:
                # Enable gradient tracking - the depth head params will create the grad fn
                features_for_depth = features_for_depth.detach().requires_grad_(True)

            # Check for NaN/Inf in features
            if torch.isnan(features_for_depth).any() or torch.isinf(features_for_depth).any():
                print(f"[WARNING] NaN/Inf in features_for_depth, clamping...")
                features_for_depth = torch.nan_to_num(features_for_depth, nan=0.0, posinf=1e6, neginf=-1e6)

            depth_pred = self.compute_depth_prediction(
                features_for_depth, t_embed, grid_size
            )

            # Check and clamp depth prediction to prevent NaN
            if torch.isnan(depth_pred).any() or torch.isinf(depth_pred).any():
                print(f"[WARNING] NaN/Inf in depth_pred, clamping...")
                depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=1e6, neginf=-1e6)

            # Resize target depth to match prediction size if needed
            target_depth = target_depth.to(device=depth_pred.device, dtype=depth_pred.dtype)

            # Check and clamp target depth
            if torch.isnan(target_depth).any() or torch.isinf(target_depth).any():
                print(f"[WARNING] NaN/Inf in target_depth, clamping...")
                target_depth = torch.nan_to_num(target_depth, nan=0.0, posinf=1e6, neginf=-1e6)

            # Depth maps have shape (1, F, H, W) but prediction has (B, 1, F, H, W)
            if target_depth.dim() == 4:
                target_depth = target_depth.unsqueeze(0)  # Add batch dim

            # Resize target depth to match prediction dimensions
            pred_f, pred_h, pred_w = depth_pred.shape[2], depth_pred.shape[3], depth_pred.shape[4]
            target_f, target_h, target_w = target_depth.shape[2], target_depth.shape[3], target_depth.shape[4]

            if (pred_f, pred_h, pred_w) != (target_f, target_h, target_w):
                # Use 3D interpolation to resize target to match prediction
                target_depth = F.interpolate(
                    target_depth,
                    size=(pred_f, pred_h, pred_w),
                    mode='trilinear',
                    align_corners=False
                )

            # Normalize both predictions and targets to similar scales for stable training
            # Use scale-invariant approach: normalize by mean of target
            target_mean = target_depth.abs().mean().clamp(min=1e-6)
            target_depth_norm = target_depth / target_mean
            depth_pred_norm = depth_pred / target_mean  # Use same scale for prediction

            # Compute depth loss on normalized values
            depth_loss = compute_depth_loss(
                depth_pred_norm, target_depth_norm,
                loss_type=self.depth_loss_type
            )

            # Final NaN check on loss
            if torch.isnan(depth_loss) or torch.isinf(depth_loss):
                print(f"[WARNING] NaN/Inf depth_loss detected!")
                print(f"  depth_pred range: [{depth_pred.min().item():.4f}, {depth_pred.max().item():.4f}]")
                print(f"  target_depth range: [{target_depth.min().item():.4f}, {target_depth.max().item():.4f}]")
                print(f"  target_mean: {target_mean.item():.4f}")
                # Replace with a small valid loss to continue training
                depth_loss = torch.tensor(0.0, device=depth_pred.device, dtype=depth_pred.dtype, requires_grad=True)

            loss_dict["depth_loss"] = depth_loss.detach().item()
            loss_dict["depth_loss_weighted"] = (self.depth_loss_weight * depth_loss).detach().item()

            weighted_depth_loss = self.depth_loss_weight * depth_loss
            if total_loss is None:
                total_loss = weighted_depth_loss
            else:
                total_loss = total_loss + weighted_depth_loss

        # Safety check: ensure we have some loss to backpropagate
        if total_loss is None:
            raise RuntimeError(
                f"No loss computed for training mode '{self.training_mode}'. "
                f"This may happen if target data (motion vectors or depth maps) is missing."
            )

        # Add total loss to dict
        loss_dict["total_loss"] = total_loss.detach().item()

        if return_loss_dict:
            return total_loss, loss_dict
        return total_loss

    def get_trainable_parameters(self):
        """Get all trainable parameters including motion and depth heads."""
        params = list(super().get_trainable_parameters())
        if self.enable_motion and hasattr(self, 'motion_head'):
            params.extend(self.motion_head.parameters())
        if self.enable_depth and hasattr(self, 'depth_head'):
            params.extend(self.depth_head.parameters())
        return params

    def trainable_modules(self):
        """
        Override to include motion and depth head parameters in optimizer.
        This method is called by the training loop to create the optimizer.
        """
        # Get base trainable modules from parent class
        try:
            base_params = list(super().trainable_modules())
        except (TypeError, AttributeError):
            # Parent class might not have this method or might return a generator
            base_params = []

        # Add motion head parameters if enabled
        if self.enable_motion and hasattr(self, 'motion_head'):
            for param in self.motion_head.parameters():
                if param.requires_grad:
                    base_params.append(param)

        # Add depth head parameters if enabled
        if self.enable_depth and hasattr(self, 'depth_head'):
            for param in self.depth_head.parameters():
                if param.requires_grad:
                    base_params.append(param)

        return base_params


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
    """
    Create a data operator for loading image sequences with motion vectors and depth maps.
    """
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


def motion_aware_wan_parser():
    """Create argument parser with motion-specific arguments."""
    parser = wan_parser()

    # Motion-specific arguments
    parser.add_argument(
        "--motion_channels",
        type=int,
        default=4,
        help="Number of motion vector channels (default: 4)",
    )
    parser.add_argument(
        "--motion_loss_weight",
        type=float,
        default=0.1,
        help="Weight for motion vector loss (default: 0.1)",
    )
    parser.add_argument(
        "--motion_loss_type",
        type=str,
        default="mse",
        choices=["mse", "l1", "smooth_l1"],
        help="Type of motion loss (default: mse)",
    )
    parser.add_argument(
        "--normalize_motion",
        action="store_true",
        help="Normalize motion vectors by image dimensions",
    )
    parser.add_argument(
        "--motion_scale",
        type=float,
        default=1.0,
        help="Scale factor for motion vectors (default: 1.0)",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="lora",
        choices=["lora", "motion_last", "depth_last", "both_last", "motion_only", "depth_only"],
        help="""Training mode:
            lora - LoRA + motion head + depth head (full adaptation)
            motion_last - motion head + DiT last layer only
            depth_last - depth head + DiT last layer only
            both_last - motion head + depth head + DiT last layer
            motion_only - motion head only (no video improvement)
            depth_only - depth head only (no video improvement)
        """,
    )

    # Depth-specific arguments
    parser.add_argument(
        "--depth_loss_weight",
        type=float,
        default=0.1,
        help="Weight for depth prediction loss (default: 0.1)",
    )
    parser.add_argument(
        "--depth_loss_type",
        type=str,
        default="mse",
        choices=["mse", "l1", "smooth_l1", "log_l1"],
        help="Type of depth loss (default: mse)",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Scale factor for depth values (default: 1.0)",
    )
    parser.add_argument(
        "--normalize_depth",
        action="store_true",
        help="Normalize depth values to [0, 1] range",
    )

    # Wandb arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="wan-motion-depth-training",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (team or username)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log to wandb every N steps",
    )

    return parser


def launch_training_task_with_wandb(
    dataset: torch.utils.data.Dataset,
    model: MotionAwareWanTrainingModule,
    model_logger: ModelLogger,
    args,
):
    """
    Launch training task with wandb logging support.

    This is a custom version of launch_training_task that adds wandb logging
    for individual losses (noise, motion, depth).
    """
    # Extract training parameters
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    gradient_accumulation_steps = args.gradient_accumulation_steps
    find_unused_parameters = args.find_unused_parameters

    # Initialize wandb if enabled
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "training_mode": args.training_mode,
            "motion_loss_weight": args.motion_loss_weight,
            "motion_loss_type": args.motion_loss_type,
            "depth_loss_weight": args.depth_loss_weight,
            "depth_loss_type": args.depth_loss_type,
            "motion_channels": args.motion_channels,
            "lora_rank": args.lora_rank if hasattr(args, 'lora_rank') else None,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
        }

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Training loop config
    global_step = 0
    log_every_n_steps = args.log_every_n_steps

    # Initialize wandb on main process only
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wandb_config,
        )
        print(f"=" * 50)
        print(f"Wandb initialized successfully!")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run name: {wandb.run.name}")
        print(f"  Run URL: {wandb.run.get_url()}")
        print(f"  Log every {log_every_n_steps} steps")
        print(f"=" * 50)
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: --use_wandb specified but wandb is not installed!")
        print("Install with: pip install wandb")
    elif args.use_wandb and not accelerator.is_main_process:
        pass  # Wandb only on main process, this is expected

    for epoch_id in range(num_epochs):
        epoch_losses = {
            "noise_loss": [],
            "motion_loss": [],
            "depth_loss": [],
            "total_loss": [],
        }

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id + 1}/{num_epochs}", disable=not accelerator.is_main_process)

        for data in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Forward pass with loss dict
                # Access the underlying module to ensure return_loss_dict works correctly
                unwrapped_model = accelerator.unwrap_model(model)

                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    result = unwrapped_model({}, inputs=data, return_loss_dict=True)
                else:
                    result = unwrapped_model(data, return_loss_dict=True)

                # Handle return value - should be (loss, loss_dict) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    loss, loss_dict = result
                else:
                    # Fallback if return_loss_dict didn't work
                    loss = result
                    loss_dict = {"total_loss": loss.detach().item(), "noise_loss": loss.detach().item()}

                # Debug: verify loss requires grad
                if not loss.requires_grad:
                    # Print debug info once
                    if global_step == 0:
                        print(f"\n[DEBUG] Loss does not require grad!")
                        print(f"  Training mode: {args.training_mode}")
                        print(f"  Loss value: {loss.item()}")
                        # Check if heads have trainable params
                        if hasattr(unwrapped_model, 'depth_head'):
                            depth_params = sum(p.numel() for p in unwrapped_model.depth_head.parameters() if p.requires_grad)
                            print(f"  Depth head trainable params: {depth_params}")
                        if hasattr(unwrapped_model, 'motion_head'):
                            motion_params = sum(p.numel() for p in unwrapped_model.motion_head.parameters() if p.requires_grad)
                            print(f"  Motion head trainable params: {motion_params}")
                    # Create a dummy grad-requiring loss to avoid crash
                    # This is a workaround - the real fix is to ensure proper gradient flow
                    dummy_param = None
                    if hasattr(unwrapped_model, 'depth_head'):
                        for p in unwrapped_model.depth_head.parameters():
                            if p.requires_grad:
                                dummy_param = p
                                break
                    if hasattr(unwrapped_model, 'motion_head') and dummy_param is None:
                        for p in unwrapped_model.motion_head.parameters():
                            if p.requires_grad:
                                dummy_param = p
                                break
                    if dummy_param is not None:
                        # Add a small regularization term to ensure gradients
                        loss = loss + 0.0 * dummy_param.sum()

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()

                global_step += 1

                # Collect losses for averaging
                epoch_losses["noise_loss"].append(loss_dict.get("noise_loss", 0))
                epoch_losses["total_loss"].append(loss_dict.get("total_loss", 0))
                if "motion_loss" in loss_dict:
                    epoch_losses["motion_loss"].append(loss_dict["motion_loss"])
                if "depth_loss" in loss_dict:
                    epoch_losses["depth_loss"].append(loss_dict["depth_loss"])

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "noise": f"{loss_dict['noise_loss']:.4f}",
                    "motion": f"{loss_dict.get('motion_loss', 0):.4f}",
                    "depth": f"{loss_dict.get('depth_loss', 0):.4f}",
                })

                # Log to wandb
                if use_wandb and accelerator.is_main_process and global_step % log_every_n_steps == 0:
                    log_data = {
                        "train/total_loss": loss_dict["total_loss"],
                        "train/noise_loss": loss_dict["noise_loss"],
                        "train/noise_loss_weighted": loss_dict.get("noise_loss_weighted", loss_dict["noise_loss"]),
                        "train/timestep": loss_dict.get("timestep", 0),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch_id,
                        "train/global_step": global_step,
                    }

                    if "motion_loss" in loss_dict:
                        log_data["train/motion_loss"] = loss_dict["motion_loss"]
                        log_data["train/motion_loss_weighted"] = loss_dict.get("motion_loss_weighted", loss_dict["motion_loss"])

                    if "depth_loss" in loss_dict:
                        log_data["train/depth_loss"] = loss_dict["depth_loss"]
                        log_data["train/depth_loss_weighted"] = loss_dict.get("depth_loss_weighted", loss_dict["depth_loss"])

                    wandb.log(log_data, step=global_step)

                    # Debug output (first few logs)
                    if global_step <= log_every_n_steps * 3:
                        print(f"[Step {global_step}] Logged to wandb: total={loss_dict['total_loss']:.4f}, "
                              f"noise={loss_dict['noise_loss']:.4f}, "
                              f"motion={loss_dict.get('motion_loss', 0):.4f}, "
                              f"depth={loss_dict.get('depth_loss', 0):.4f}")

                # Save checkpoint
                model_logger.on_step_end(accelerator, model, save_steps)

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

            wandb.log(epoch_summary, step=global_step)

        # Save epoch checkpoint
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    # Final save
    model_logger.on_training_end(accelerator, model, save_steps)

    # Finish wandb
    if use_wandb and accelerator.is_main_process:
        wandb.finish()
        print("Wandb run finished.")


if __name__ == "__main__":
    parser = motion_aware_wan_parser()
    args = parser.parse_args()

    # Print training configuration
    print("=" * 60)
    print("WAN Motion/Depth Training")
    print("=" * 60)
    print(f"Training mode: {args.training_mode}")
    print(f"Motion loss weight: {args.motion_loss_weight}")
    print(f"Depth loss weight: {args.depth_loss_weight}")
    print(f"Wandb enabled: {args.use_wandb}")
    if args.use_wandb:
        print(f"  Project: {args.wandb_project}")
        print(f"  Run name: {args.wandb_run_name or 'auto'}")
    print("=" * 60)

    # Determine which heads are enabled based on training mode
    load_motion = args.training_mode in ["lora", "motion_last", "both_last", "motion_only"]
    load_depth = args.training_mode in ["lora", "depth_last", "both_last", "depth_only"]

    # Create dataset with motion vector and depth loading
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
            load_depth=load_depth,
        ),
    )

    # Create motion and depth aware training module
    model = MotionAwareWanTrainingModule(
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
        training_mode=args.training_mode,
        depth_loss_weight=args.depth_loss_weight,
        depth_loss_type=args.depth_loss_type,
    )

    # Create model logger
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)

    # Launch training with wandb support
    launch_training_task_with_wandb(dataset, model, model_logger, args=args)
