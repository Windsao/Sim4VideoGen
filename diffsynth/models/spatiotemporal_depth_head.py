"""
Spatio-Temporal Depth Head for WAN Video DiT.

This module implements a Video-Depth-Anything style spatio-temporal head
for predicting temporally consistent depth maps from video features.

The key innovations from Video-Depth-Anything:
1. Multi-scale feature processing with temporal attention at each scale
2. Cross-frame temporal attention for depth consistency
3. DPT-style feature fusion with progressive refinement

Reference: Video Depth Anything (arXiv:2501.12375)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange


def get_sinusoidal_positional_encoding(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Generate sinusoidal positional encodings for temporal positions."""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) *
                         (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def precompute_temporal_freqs(dim: int, max_frames: int = 256, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequencies for temporal dimension."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_frames)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_temporal_rope(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply rotary position embeddings for temporal attention."""
    # x: (B*H*W, T, D)
    # freqs: (T, D//2) complex
    B_HW, T, D = x.shape
    head_dim = D // num_heads

    x = x.view(B_HW, T, num_heads, head_dim)
    x_complex = torch.view_as_complex(x.float().reshape(B_HW, T, num_heads, -1, 2))

    # Broadcast freqs to match head dimension
    freqs = freqs[:T, :head_dim//2].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim//2)
    freqs = freqs.to(x.device)

    x_out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_out.view(B_HW, T, D).to(x.dtype)


class GEGLU(nn.Module):
    """Gated GELU activation function."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention(nn.Module):
    """
    Temporal attention module for cross-frame feature aggregation.

    Processes spatial positions independently, applying attention across
    the temporal dimension for each spatial location.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        pos_embed_type: str = "rope",  # "rope" or "ape" (absolute positional encoding)
        max_frames: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        inner_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_embed_type = pos_embed_type

        # Q, K, V projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # Positional encoding
        if pos_embed_type == "rope":
            self.register_buffer("temporal_freqs", precompute_temporal_freqs(self.head_dim, max_frames))
        elif pos_embed_type == "ape":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Features of shape (B, C, T, H, W)
            num_frames: Number of frames T
            cached_kv: Optional cached key-value for streaming inference

        Returns:
            Output features (B, C, T, H, W) and updated cached_kv
        """
        B, C, T, H, W = x.shape

        # Reshape: (B, C, T, H, W) -> (B*H*W, T, C)
        x = rearrange(x, 'b c t h w -> (b h w) t c')

        # Add positional encoding if using APE
        if self.pos_embed_type == "ape":
            x = x + self.pos_embed[:, :T, :].to(x.dtype)

        # Compute Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Apply RoPE if using rotary embeddings
        if self.pos_embed_type == "rope":
            q = apply_temporal_rope(q, self.temporal_freqs, self.num_heads)
            k = apply_temporal_rope(k, self.temporal_freqs, self.num_heads)

        # Reshape for multi-head attention: (B*H*W, T, num_heads, head_dim)
        q = rearrange(q, 'b t (n d) -> b n t d', n=self.num_heads)
        k = rearrange(k, 'b t (n d) -> b n t d', n=self.num_heads)
        v = rearrange(v, 'b t (n d) -> b n t d', n=self.num_heads)

        # Handle cached KV for streaming
        if cached_kv is not None:
            k = torch.cat([cached_kv[0], k], dim=2)
            v = torch.cat([cached_kv[1], v], dim=2)
        new_cached_kv = (k, v)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)

        # Reshape back: (B*H*W, num_heads, T, head_dim) -> (B*H*W, T, dim)
        attn = rearrange(attn, 'b n t d -> b t (n d)')

        # Output projection
        out = self.to_out(attn)

        # Reshape back to spatial: (B*H*W, T, C) -> (B, C, T, H, W)
        out = rearrange(out, '(b h w) t c -> b c t h w', b=B, h=H, w=W)

        return out, new_cached_kv


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block with temporal attention and feed-forward network.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.0,
        pos_embed_type: str = "rope",
        max_frames: int = 256,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pos_embed_type=pos_embed_type,
            max_frames=max_frames,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Features of shape (B, C, T, H, W)
            num_frames: Number of frames
            cached_kv: Optional cached key-value

        Returns:
            Output features and updated cached_kv
        """
        B, C, T, H, W = x.shape

        # Apply LayerNorm in channel dimension
        x_norm = rearrange(x, 'b c t h w -> b t h w c')
        x_norm = self.norm1(x_norm)
        x_norm = rearrange(x_norm, 'b t h w c -> b c t h w')

        # Temporal attention with residual
        attn_out, new_cached_kv = self.attn(x_norm, num_frames, cached_kv)
        x = x + attn_out

        # Feed-forward with residual
        x_norm = rearrange(x, 'b c t h w -> b t h w c')
        x_norm = self.norm2(x_norm)
        ff_out = self.ff(x_norm)
        ff_out = rearrange(ff_out, 'b t h w c -> b c t h w')
        x = x + ff_out

        return x, new_cached_kv


class TemporalModule(nn.Module):
    """
    Temporal processing module inspired by Video-Depth-Anything.

    Wraps temporal transformer blocks and handles the interface with
    the multi-scale feature pyramid.
    """
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        num_blocks: int = 1,
        ff_mult: int = 4,
        dropout: float = 0.0,
        pos_embed_type: str = "rope",
        max_frames: int = 256,
        zero_init: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Input projection (optional, if we want to adjust channels)
        self.proj_in = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # Stack of temporal transformer blocks
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                dim=in_channels,
                num_heads=num_heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
                dropout=dropout,
                pos_embed_type=pos_embed_type,
                max_frames=max_frames,
            )
            for _ in range(num_blocks)
        ])

        # Output projection with optional zero initialization
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        if zero_init:
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        cached_hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Features of shape (B, C, T, H, W)
            num_frames: Number of frames
            cached_hidden_states: List of cached KV states for each block

        Returns:
            Output features and list of new cached states
        """
        residual = x
        x = self.proj_in(x)

        new_cached_states = []
        for i, block in enumerate(self.blocks):
            cached_kv = cached_hidden_states[i] if cached_hidden_states else None
            x, new_kv = block(x, num_frames, cached_kv)
            new_cached_states.append(new_kv)

        x = self.proj_out(x)
        x = x + residual

        return x, new_cached_states


class ConvBlock(nn.Module):
    """Convolution block with optional batch normalization."""
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FeatureFusionBlock(nn.Module):
    """Feature fusion block for combining multi-scale features."""
    def __init__(self, features: int, use_bn: bool = False):
        super().__init__()
        self.conv1 = ConvBlock(features, features, use_bn)
        self.conv2 = ConvBlock(features, features, use_bn)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = x + skip
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.out_conv(x)
        return x


class SpatioTemporalDepthHead(nn.Module):
    """
    Spatio-Temporal Depth Head inspired by Video-Depth-Anything.

    This head processes DiT features through:
    1. Multi-scale feature extraction
    2. Temporal attention at each scale level
    3. Progressive feature fusion
    4. Final depth prediction

    The temporal modules ensure depth consistency across frames by
    allowing information exchange between temporal positions.
    """
    def __init__(
        self,
        dim: int,  # DiT feature dimension
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        features: int = 256,  # Internal feature dimension for fusion
        num_temporal_heads: int = 8,
        temporal_head_dim: int = 64,
        num_temporal_blocks: int = 1,
        use_bn: bool = False,
        pos_embed_type: str = "rope",
        max_frames: int = 256,
        output_scale: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.features = features
        self.output_scale = output_scale

        # Layer norm before processing
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Modulation parameters (timestep-dependent)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

        # Project from DiT dimension to internal features
        # We process at 4 different scales inspired by DPT
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(dim, features, kernel_size=1),
            nn.Conv2d(dim, features, kernel_size=1),
            nn.Conv2d(dim, features, kernel_size=1),
            nn.Conv2d(dim, features, kernel_size=1),
        ])

        # Temporal modules at each scale
        self.temporal_modules = nn.ModuleList([
            TemporalModule(
                in_channels=features,
                num_heads=num_temporal_heads,
                head_dim=temporal_head_dim,
                num_blocks=num_temporal_blocks,
                pos_embed_type=pos_embed_type,
                max_frames=max_frames,
                zero_init=True,
            )
            for _ in range(4)
        ])

        # Spatial resize operations for multi-scale
        # Scale 0: 1x, Scale 1: 2x, Scale 2: 4x, Scale 3: 8x (relative to patch grid)
        self.resize_ops = nn.ModuleList([
            nn.Identity(),  # Scale 0: keep original
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),  # Scale 1: 2x
            nn.Sequential(
                nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
                nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
            ),  # Scale 2: 4x
            nn.Sequential(
                nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
                nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
                nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
            ),  # Scale 3: 8x
        ])

        # Feature fusion blocks (bottom-up fusion)
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(features, use_bn),
            FeatureFusionBlock(features, use_bn),
            FeatureFusionBlock(features, use_bn),
            FeatureFusionBlock(features, use_bn),
        ])

        # Output head
        self.output_conv = nn.Sequential(
            ConvBlock(features, features // 2, use_bn),
            nn.Conv2d(features // 2, 1, kernel_size=1),
        )

        # Initialize output conv with small weights
        nn.init.xavier_uniform_(self.output_conv[-1].weight, gain=0.01)
        if self.output_conv[-1].bias is not None:
            nn.init.zeros_(self.output_conv[-1].bias)

    def _reshape_features_to_spatial(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Reshape DiT features from sequence to spatial format.

        Args:
            x: Features of shape (B, F*H*W, D)
            grid_size: (F, H, W) patch grid dimensions

        Returns:
            Spatial features of shape (B*F, D, H, W)
        """
        f, h, w = grid_size
        B = x.shape[0]

        # (B, F*H*W, D) -> (B, F, H, W, D) -> (B*F, D, H, W)
        x = x.view(B, f, h, w, -1)
        x = rearrange(x, 'b f h w d -> (b f) d h w')
        return x

    def _reshape_to_video(
        self,
        x: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Reshape from batched frames to video format.

        Args:
            x: Features of shape (B*F, C, H, W)
            batch_size: Original batch size
            num_frames: Number of frames

        Returns:
            Video features of shape (B, C, F, H, W)
        """
        return rearrange(x, '(b f) c h w -> b c f h w', b=batch_size, f=num_frames)

    def _reshape_from_video(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reshape from video to batched frames format.

        Args:
            x: Video features of shape (B, C, F, H, W)

        Returns:
            Batched frame features of shape (B*F, C, H, W)
        """
        return rearrange(x, 'b c f h w -> (b f) c h w')

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        grid_size: Tuple[int, int, int],
        cached_hidden_states: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for depth prediction.

        Args:
            x: DiT features of shape (B, F*H*W, D) where S = F*H*W (sequence length)
            t_mod: Timestep modulation of shape (B, D) or (B, N, D)
            grid_size: (F, H, W) patch grid dimensions
            cached_hidden_states: Optional list of cached states for each temporal module

        Returns:
            Depth predictions of shape (B, 1, F*patch_f, H*patch_h, W*patch_w)
            List of new cached hidden states
        """
        f, h, w = grid_size
        B = x.shape[0]

        # Apply timestep modulation
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.norm(x) * (1 + scale) + shift

        # Reshape to spatial: (B, F*H*W, D) -> (B*F, D, H, W)
        x_spatial = self._reshape_features_to_spatial(x, grid_size)

        # Process through multi-scale pipeline
        # For simplicity, we use the same features at different scales
        # (In Video-Depth-Anything, they use features from different encoder layers)
        layer_outputs = []
        for i, (proj, resize) in enumerate(zip(self.proj_layers, self.resize_ops)):
            feat = proj(x_spatial)  # (B*F, features, H, W)
            # Apply resize for multi-scale
            if i > 0:
                # Downsample for coarser scales
                scale_factor = 1.0 / (2 ** i)
                feat = F.interpolate(feat, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            layer_outputs.append(feat)

        # Apply temporal attention at each scale
        new_cached_states = []
        temporal_outputs = []
        for i, (feat, temporal_module) in enumerate(zip(layer_outputs, self.temporal_modules)):
            # Get spatial dimensions
            _, C, feat_H, feat_W = feat.shape

            # Reshape to video format for temporal attention: (B*F, C, H, W) -> (B, C, F, H, W)
            feat_video = self._reshape_to_video(feat, B, f)

            # Apply temporal attention
            cached = cached_hidden_states[i] if cached_hidden_states else None
            feat_video, new_cached = temporal_module(feat_video, f, cached)
            new_cached_states.append(new_cached)

            # Reshape back: (B, C, F, H, W) -> (B*F, C, H, W)
            feat = self._reshape_from_video(feat_video)

            # Upsample back to original scale
            if i > 0:
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)

            temporal_outputs.append(feat)

        # Feature fusion (coarse to fine)
        # Start from coarsest scale and progressively fuse with finer scales
        fused = temporal_outputs[3]  # Coarsest
        for i in range(2, -1, -1):
            fused = self.fusion_blocks[i](fused, temporal_outputs[i])

        # Final depth prediction
        depth = self.output_conv(fused)  # (B*F, 1, H_out, W_out)

        # Get output spatial size
        _, _, H_out, W_out = depth.shape

        # Reshape to video format: (B*F, 1, H, W) -> (B, 1, F, H, W)
        depth = self._reshape_to_video(depth, B, f)

        # Upsample to match patch size if needed
        target_h = h * self.patch_size[1]
        target_w = w * self.patch_size[2]
        if H_out != target_h or W_out != target_w:
            depth = F.interpolate(depth, size=(f * self.patch_size[0], target_h, target_w),
                                 mode='trilinear', align_corners=True)

        return depth * self.output_scale, new_cached_states


class SpatioTemporalDepthHeadSimple(nn.Module):
    """
    Simplified Spatio-Temporal Depth Head.

    A lighter version that adds temporal attention to the existing DepthHead
    architecture without full multi-scale processing. Useful when compute
    is limited or when the DiT features already have good temporal coherence.
    """
    def __init__(
        self,
        dim: int,
        depth_channels: int = 1,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_temporal_heads: int = 8,
        temporal_head_dim: int = 64,
        num_temporal_blocks: int = 2,
        pos_embed_type: str = "rope",
        max_frames: int = 256,
        output_scale: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.depth_channels = depth_channels
        self.patch_size = patch_size
        self.output_scale = output_scale

        # Layer norm before processing
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        # Modulation parameters
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

        # Temporal attention module
        self.temporal_module = TemporalModule(
            in_channels=dim,
            num_heads=num_temporal_heads,
            head_dim=temporal_head_dim,
            num_blocks=num_temporal_blocks,
            pos_embed_type=pos_embed_type,
            max_frames=max_frames,
            zero_init=True,
        )

        # Final projection to depth
        self.head = nn.Linear(dim, depth_channels * math.prod(patch_size))

        # Initialize with small weights
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        grid_size: Tuple[int, int, int],
        cached_hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for depth prediction with temporal attention.

        Args:
            x: DiT features of shape (B, F*H*W, D)
            t_mod: Timestep modulation of shape (B, D) or (B, N, D)
            grid_size: (F, H, W) patch grid dimensions
            cached_hidden_states: Optional cached states for temporal module

        Returns:
            Depth predictions (before unpatchify) of shape (B, F*H*W, depth_channels * patch_product)
            New cached hidden states
        """
        f, h, w = grid_size
        B = x.shape[0]

        # Apply timestep modulation
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = self.norm(x) * (1 + scale) + shift

        # Reshape for temporal attention: (B, F*H*W, D) -> (B, D, F, H, W)
        x = rearrange(x, 'b (f h w) d -> b d f h w', f=f, h=h, w=w)

        # Apply temporal attention
        x, new_cached = self.temporal_module(x, f, cached_hidden_states)

        # Reshape back: (B, D, F, H, W) -> (B, F*H*W, D)
        x = rearrange(x, 'b d f h w -> b (f h w) d')

        # Project to depth
        depth = self.head(x) * self.output_scale

        return depth, new_cached

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Convert patchified depth to full resolution.

        Args:
            x: Patchified depth of shape (B, F*H*W, depth_channels * patch_product)
            grid_size: (F, H, W) patch grid dimensions

        Returns:
            Depth of shape (B, depth_channels, F*patch_f, H*patch_h, W*patch_w)
        """
        f, h, w = grid_size
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
            c=self.depth_channels
        )
