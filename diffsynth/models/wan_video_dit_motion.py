"""
WAN Video DiT Model with Motion Vector and Depth Output Heads.

This module extends the WanModel to support multiple outputs:
1. Standard noise prediction for video generation
2. Motion vector prediction for physics-aware training
3. Depth prediction for 3D structure awareness

Motion vectors are 4-channel tensors (dx, dy, channel2, channel3) per pixel per frame.
Depth is a single-channel tensor representing distance to camera per pixel per frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange

from .wan_video_dit import (
    WanModel, Head, RMSNorm, DiTBlock, MLP,
    precompute_freqs_cis_3d, sinusoidal_embedding_1d,
    flash_attention, modulate, rope_apply
)


class MotionVectorHead(nn.Module):
    """
    Head for predicting motion vectors from DiT features.

    The motion vectors have shape (B, 4, F-1, H, W) representing:
    - Channel 0: dx (horizontal motion)
    - Channel 1: dy (vertical motion)
    - Channel 2-3: additional motion information (depth/validity)

    Note: F-1 because motion vectors represent inter-frame motion.
    """

    def __init__(
        self,
        dim: int,
        motion_channels: int = 4,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.motion_channels = motion_channels
        self.patch_size = patch_size
        self.output_scale = output_scale

        # Layer norm before projection
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Project to motion vector space
        # Output: motion_channels * patch_size product
        self.head = nn.Linear(dim, motion_channels * math.prod(patch_size))

        # Modulation parameters (similar to main head)
        # Use smaller initialization to prevent large outputs
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

        # Initialize head with small weights for stability
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: DiT features of shape (B, S, D) where S = F*H*W (sequence length)
            t_mod: Time embedding modulation of shape (B, D)

        Returns:
            Motion vector predictions
        """
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)

        return x * self.output_scale


class DepthHead(nn.Module):
    """
    Head for predicting depth (distance to camera) from DiT features.

    The depth output has shape (B, 1, F, H, W) representing distance to camera
    for each pixel in each frame.
    """

    def __init__(
        self,
        dim: int,
        depth_channels: int = 1,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth_channels = depth_channels
        self.patch_size = patch_size
        self.output_scale = output_scale

        # Layer norm before projection
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Project to depth space
        # Output: depth_channels * patch_size product
        self.head = nn.Linear(dim, depth_channels * math.prod(patch_size))

        # Modulation parameters (similar to main head)
        # Use smaller initialization to prevent large outputs
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

        # Initialize head with small weights for stability
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: DiT features of shape (B, S, D) where S = F*H*W (sequence length)
            t_mod: Time embedding modulation of shape (B, D)

        Returns:
            Depth predictions of shape (B, S, depth_channels * patch_product)
        """
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + scale) + shift)

        return x * self.output_scale


class WanModelWithMotion(WanModel):
    """
    Extended WAN model that outputs both noise prediction and motion vectors.

    This model adds a parallel motion vector head that predicts optical flow-like
    motion vectors for physics-aware video generation training.
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        # Motion vector specific parameters
        motion_channels: int = 4,
        motion_output_scale: float = 1.0,
        enable_motion_head: bool = True,
    ):
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=has_image_input,
            has_image_pos_emb=has_image_pos_emb,
            has_ref_conv=has_ref_conv,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

        self.enable_motion_head = enable_motion_head
        self.motion_channels = motion_channels

        if enable_motion_head:
            # Add motion vector head
            self.motion_head = MotionVectorHead(
                dim=dim,
                motion_channels=motion_channels,
                patch_size=patch_size,
                eps=eps,
                output_scale=motion_output_scale,
            )

    def unpatchify_motion(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Unpatchify motion vector predictions.

        Args:
            x: Patchified motion vectors of shape (B, F*H*W, C*patch_product)
            grid_size: (F, H, W) grid dimensions

        Returns:
            Motion vectors of shape (B, motion_channels, F, H*patch_h, W*patch_w)
        """
        f, h, w = grid_size
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
            c=self.motion_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_motion_vectors: bool = False,
        **kwargs,
    ):
        """
        Forward pass with optional motion vector output.

        Args:
            x: Input latents
            timestep: Diffusion timestep
            context: Text embeddings
            clip_feature: CLIP image features (optional)
            y: Additional conditioning (optional)
            use_gradient_checkpointing: Enable gradient checkpointing
            use_gradient_checkpointing_offload: Offload checkpoints to CPU
            return_motion_vectors: If True, return (noise_pred, motion_vectors)

        Returns:
            If return_motion_vectors is False: noise prediction
            If return_motion_vectors is True: (noise_pred, motion_vectors)
        """
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        # Main output (noise prediction)
        noise_pred = self.head(x, t)
        noise_pred = self.unpatchify(noise_pred, (f, h, w))

        if return_motion_vectors and self.enable_motion_head:
            # Motion vector prediction
            motion_pred = self.motion_head(x, t)
            motion_pred = self.unpatchify_motion(motion_pred, (f, h, w))
            return noise_pred, motion_pred

        return noise_pred

    @staticmethod
    def from_base_model(base_model: WanModel, motion_channels: int = 4, motion_output_scale: float = 1.0):
        """
        Create a WanModelWithMotion from an existing WanModel.

        This copies all weights from the base model and initializes
        the motion head with new random weights.

        Args:
            base_model: Pre-trained WanModel
            motion_channels: Number of motion vector channels
            motion_output_scale: Scale factor for motion output

        Returns:
            WanModelWithMotion with copied weights
        """
        # Get config from base model
        config = {
            'dim': base_model.dim,
            'in_dim': base_model.in_dim,
            'ffn_dim': base_model.blocks[0].ffn_dim if hasattr(base_model.blocks[0], 'ffn_dim') else base_model.blocks[0].ffn[0].out_features,
            'out_dim': base_model.head.head.out_features // math.prod(base_model.patch_size),
            'text_dim': base_model.text_embedding[0].in_features,
            'freq_dim': base_model.freq_dim,
            'eps': 1e-6,
            'patch_size': base_model.patch_size,
            'num_heads': base_model.blocks[0].num_heads,
            'num_layers': len(base_model.blocks),
            'has_image_input': base_model.has_image_input,
            'motion_channels': motion_channels,
            'motion_output_scale': motion_output_scale,
            'enable_motion_head': True,
        }

        # Create new model
        model = WanModelWithMotion(**config)

        # Copy base model state dict (excluding motion head)
        base_state_dict = base_model.state_dict()
        model_state_dict = model.state_dict()

        # Copy matching keys
        for key in base_state_dict:
            if key in model_state_dict:
                model_state_dict[key] = base_state_dict[key]

        model.load_state_dict(model_state_dict)

        return model


def compute_motion_loss(
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    loss_type: str = "mse",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute loss between predicted and target motion vectors.

    Args:
        pred_motion: Predicted motion vectors (B, C, F, H, W)
        target_motion: Target motion vectors (B, C, F, H, W)
        loss_type: Type of loss ("mse", "l1", "smooth_l1")
        mask: Optional mask for valid motion regions (B, 1, F, H, W)

    Returns:
        Scalar loss value
    """
    if loss_type == "mse":
        loss = F.mse_loss(pred_motion, target_motion, reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(pred_motion, target_motion, reduction='none')
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred_motion, target_motion, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if mask is not None:
        # Apply mask and compute mean only over valid regions
        loss = loss * mask
        loss = loss.sum() / (mask.sum() * pred_motion.shape[1] + 1e-8)
    else:
        loss = loss.mean()

    return loss


def warp_depth(
    depth: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Warp depth from time t to t+1 using optical flow.

    This performs backward warping: for each pixel in t+1, look up the depth
    at the corresponding position in t (determined by the flow).

    Args:
        depth: Depth at time t, shape (B, 1, H, W)
        flow: Optical flow from t to t+1, shape (B, 2, H, W) where channels are (dx, dy)
        mode: Interpolation mode for grid_sample ('bilinear' or 'nearest')
        padding_mode: Padding mode for out-of-bound values ('border', 'zeros', 'reflection')

    Returns:
        Warped depth at time t+1, shape (B, 1, H, W)
    """
    B, _, H, W = depth.shape

    # Create base grid
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing='ij'
    )
    grid = torch.stack([x, y], dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]

    # For backward warping, we subtract the flow
    # (where does each pixel in t+1 come from in t)
    # If flow represents forward flow (t -> t+1), we use it directly
    # The flow tells us where pixel at (x, y) moves to, so to find
    # where pixel (x, y) in t+1 comes from, we need the inverse
    # For simplicity, we assume the flow is small enough that
    # subtracting works approximately
    new_grid = grid - flow[:, :2]  # Only use dx, dy channels

    # Normalize to [-1, 1] for grid_sample
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0  # x
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0  # y

    # Permute for grid_sample: [B, H, W, 2]
    new_grid = new_grid.permute(0, 2, 3, 1)

    # Warp depth
    warped_depth = F.grid_sample(
        depth, new_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )

    return warped_depth


def compute_warp_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    motion_flow: torch.Tensor,
    loss_type: str = "mse",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute temporal consistency loss by warping predicted depth.

    For each frame t (except the last), warp depth_t using flow_t to get
    predicted depth_{t+1}, then compare with actual depth_{t+1}.

    Args:
        pred_depth: Predicted depth (B, 1, T, H, W)
        target_depth: Target depth (B, 1, T, H, W)
        motion_flow: Motion flow GT (B, C, T-1, H, W) where C >= 2 (dx, dy, ...)
        loss_type: Type of loss ("mse", "l1", "smooth_l1")
        mask: Optional mask for valid regions (B, 1, T-1, H, W)

    Returns:
        Scalar warp loss value
    """
    B, _, T, H, W = pred_depth.shape
    T_flow = motion_flow.shape[2]

    # Flow has T-1 frames (inter-frame motion)
    # We can compute warp loss for frames 0 to T-2
    num_pairs = min(T - 1, T_flow)

    if num_pairs <= 0:
        # Not enough frames for warp loss
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

    total_loss = 0.0

    for t in range(num_pairs):
        # Get depth at time t
        depth_t = pred_depth[:, :, t, :, :]  # (B, 1, H, W)

        # Get flow from t to t+1
        flow_t = motion_flow[:, :2, t, :, :]  # (B, 2, H, W), only dx, dy

        # Get target depth at t+1
        depth_t1_target = target_depth[:, :, t + 1, :, :]  # (B, 1, H, W)

        # Warp depth_t to t+1 using flow
        depth_t1_warped = warp_depth(depth_t, flow_t)

        # Compute loss between warped depth and target depth at t+1
        if loss_type == "mse":
            frame_loss = F.mse_loss(depth_t1_warped, depth_t1_target, reduction='none')
        elif loss_type == "l1":
            frame_loss = F.l1_loss(depth_t1_warped, depth_t1_target, reduction='none')
        elif loss_type == "smooth_l1":
            frame_loss = F.smooth_l1_loss(depth_t1_warped, depth_t1_target, reduction='none')
        else:
            frame_loss = F.mse_loss(depth_t1_warped, depth_t1_target, reduction='none')

        if mask is not None and t < mask.shape[2]:
            frame_mask = mask[:, :, t, :, :]
            frame_loss = frame_loss * frame_mask
            frame_loss = frame_loss.sum() / (frame_mask.sum() + 1e-8)
        else:
            frame_loss = frame_loss.mean()

        total_loss = total_loss + frame_loss

    # Average over all frame pairs
    return total_loss / num_pairs


def compute_depth_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    loss_type: str = "mse",
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = False,
) -> torch.Tensor:
    """
    Compute loss between predicted and target depth maps.

    Args:
        pred_depth: Predicted depth (B, 1, F, H, W)
        target_depth: Target depth (B, 1, F, H, W)
        loss_type: Type of loss ("mse", "l1", "smooth_l1", "log_l1")
        mask: Optional mask for valid depth regions (B, 1, F, H, W)
        scale_invariant: If True, use scale-invariant loss

    Returns:
        Scalar loss value
    """
    if scale_invariant:
        # Scale-invariant depth loss (useful when absolute scale is uncertain)
        # Compute in log space
        pred_log = torch.log(pred_depth.clamp(min=1e-8))
        target_log = torch.log(target_depth.clamp(min=1e-8))
        diff = pred_log - target_log

        if mask is not None:
            diff = diff * mask
            n_valid = mask.sum().clamp(min=1)
            loss = (diff ** 2).sum() / n_valid - 0.5 * (diff.sum() ** 2) / (n_valid ** 2)
        else:
            n = pred_depth.numel()
            loss = (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)
        return loss

    if loss_type == "mse":
        loss = F.mse_loss(pred_depth, target_depth, reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(pred_depth, target_depth, reduction='none')
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred_depth, target_depth, reduction='none')
    elif loss_type == "log_l1":
        # L1 loss in log space - good for depth
        pred_log = torch.log(pred_depth.clamp(min=1e-8))
        target_log = torch.log(target_depth.clamp(min=1e-8))
        loss = F.l1_loss(pred_log, target_log, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if mask is not None:
        # Apply mask and compute mean only over valid regions
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()

    return loss
