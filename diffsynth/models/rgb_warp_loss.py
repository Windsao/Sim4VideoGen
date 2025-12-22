"""
RGB Warp Loss for Temporal Consistency.

This module provides functions to compute RGB warp loss (photometric loss)
for enforcing temporal consistency in video predictions. It uses optical flow
to warp RGB frames and compute the difference between warped and target frames.

Inspired by classical photometric loss used in:
- Optical flow estimation (FlowNet, PWC-Net)
- Self-supervised depth estimation (Monodepth, SfMLearner)
- Video prediction models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def warp_image(
    image: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Warp image from time t to t+1 using optical flow.

    This performs backward warping: for each pixel in t+1, look up the value
    at the corresponding position in t (determined by the flow).

    Args:
        image: Image at time t, shape (B, C, H, W) where C can be any number of channels
        flow: Optical flow from t to t+1, shape (B, 2, H, W) where channels are (dx, dy)
        mode: Interpolation mode for grid_sample ('bilinear' or 'nearest')
        padding_mode: Padding mode for out-of-bound values ('border', 'zeros', 'reflection')

    Returns:
        Warped image at time t+1, shape (B, C, H, W)
    """
    B, C, H, W = image.shape

    # Create base grid
    y, x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=image.dtype),
        torch.arange(W, device=image.device, dtype=image.dtype),
        indexing='ij'
    )
    grid = torch.stack([x, y], dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]

    # For backward warping, we subtract the flow
    # The flow tells us where pixel at (x, y) moves to, so to find
    # where pixel (x, y) in t+1 comes from, we use backward flow
    new_grid = grid - flow[:, :2]  # Only use dx, dy channels

    # Normalize to [-1, 1] for grid_sample
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0  # x
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0  # y

    # Permute for grid_sample: [B, H, W, 2]
    new_grid = new_grid.permute(0, 2, 3, 1)

    # Warp image
    warped_image = F.grid_sample(
        image, new_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )

    return warped_image


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM is more robust than MSE/L1 for image comparison as it considers
    luminance, contrast, and structure.

    Args:
        pred: Predicted image (B, C, H, W), values should be in [0, 1]
        target: Target image (B, C, H, W), values should be in [0, 1]
        window_size: Size of the Gaussian window (default: 11)
        size_average: If True, return mean SSIM, else return SSIM map
        C1: Constant for luminance stability
        C2: Constant for contrast stability

    Returns:
        SSIM value (scalar if size_average=True, else (B, C, H, W))
    """
    B, C, H, W = pred.shape

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()

    # Create 2D window
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(C, 1, window_size, window_size).contiguous()
    window = window.to(pred.device).type_as(pred)

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


def compute_charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """
    Compute Charbonnier loss (smooth L1 alternative).

    Charbonnier loss: sqrt((pred - target)^2 + epsilon^2)
    This is differentiable everywhere and more robust to outliers than L2.

    Args:
        pred: Predicted tensor
        target: Target tensor
        epsilon: Small constant for numerical stability

    Returns:
        Charbonnier loss value
    """
    diff = pred - target
    loss = torch.sqrt(diff * diff + epsilon * epsilon)
    return loss.mean()


def compute_rgb_warp_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    motion_flow: torch.Tensor,
    loss_type: str = "l1",
    mask: Optional[torch.Tensor] = None,
    use_ssim: bool = False,
    ssim_weight: float = 0.85,
) -> torch.Tensor:
    """
    Compute temporal consistency loss by warping predicted RGB frames.

    For each frame t (except the last), warp rgb_t using flow_t to get
    predicted rgb_{t+1}, then compare with actual rgb_{t+1}.

    Args:
        pred_rgb: Predicted RGB frames (B, 3, T, H, W)
        target_rgb: Target RGB frames (B, 3, T, H, W)
        motion_flow: Motion flow GT (B, C, T-1, H, W) where C >= 2 (dx, dy, ...)
        loss_type: Type of loss ("mse", "l1", "smooth_l1", "charbonnier")
        mask: Optional mask for valid regions (B, 1, T-1, H, W)
        use_ssim: If True, combine pixel loss with SSIM loss
        ssim_weight: Weight for SSIM when use_ssim=True (pixel_weight = 1 - ssim_weight)

    Returns:
        Scalar warp loss value
    """
    B, C, T, H, W = pred_rgb.shape
    T_flow = motion_flow.shape[2]

    # Flow has T-1 frames (inter-frame motion)
    num_pairs = min(T - 1, T_flow)

    if num_pairs <= 0:
        return torch.tensor(0.0, device=pred_rgb.device, dtype=pred_rgb.dtype)

    total_loss = 0.0

    for t in range(num_pairs):
        # Get RGB at time t
        rgb_t = pred_rgb[:, :, t, :, :]  # (B, 3, H, W)

        # Get flow from t to t+1
        flow_t = motion_flow[:, :2, t, :, :]  # (B, 2, H, W), only dx, dy

        # Get target RGB at t+1
        rgb_t1_target = target_rgb[:, :, t + 1, :, :]  # (B, 3, H, W)

        # Warp rgb_t to t+1 using flow
        rgb_t1_warped = warp_image(rgb_t, flow_t)

        # Compute pixel-wise loss
        if loss_type == "mse":
            pixel_loss = F.mse_loss(rgb_t1_warped, rgb_t1_target, reduction='none')
        elif loss_type == "l1":
            pixel_loss = F.l1_loss(rgb_t1_warped, rgb_t1_target, reduction='none')
        elif loss_type == "smooth_l1":
            pixel_loss = F.smooth_l1_loss(rgb_t1_warped, rgb_t1_target, reduction='none')
        elif loss_type == "charbonnier":
            diff = rgb_t1_warped - rgb_t1_target
            pixel_loss = torch.sqrt(diff * diff + 1e-6)
        else:
            pixel_loss = F.l1_loss(rgb_t1_warped, rgb_t1_target, reduction='none')

        # Apply mask if provided
        if mask is not None and t < mask.shape[2]:
            frame_mask = mask[:, :, t, :, :]  # (B, 1, H, W)
            # Expand mask to match RGB channels
            frame_mask = frame_mask.expand(-1, C, -1, -1)
            pixel_loss = pixel_loss * frame_mask
            frame_loss = pixel_loss.sum() / (frame_mask.sum() + 1e-8)
        else:
            frame_loss = pixel_loss.mean()

        # Optionally add SSIM loss
        if use_ssim:
            # Normalize to [0, 1] range for SSIM if needed
            rgb_warped_norm = (rgb_t1_warped - rgb_t1_warped.min()) / (rgb_t1_warped.max() - rgb_t1_warped.min() + 1e-8)
            rgb_target_norm = (rgb_t1_target - rgb_t1_target.min()) / (rgb_t1_target.max() - rgb_t1_target.min() + 1e-8)

            ssim_loss = 1.0 - compute_ssim(rgb_warped_norm, rgb_target_norm)

            # Combine losses
            frame_loss = ssim_weight * ssim_loss + (1.0 - ssim_weight) * frame_loss

        total_loss = total_loss + frame_loss

    return total_loss / num_pairs


def compute_bidirectional_rgb_warp_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute bidirectional RGB warp loss for stronger temporal consistency.

    Computes both forward warping (t -> t+1) and backward warping (t+1 -> t)
    if backward flow is provided.

    Args:
        pred_rgb: Predicted RGB frames (B, 3, T, H, W)
        target_rgb: Target RGB frames (B, 3, T, H, W)
        forward_flow: Forward flow (B, 2, T-1, H, W)
        backward_flow: Backward flow (B, 2, T-1, H, W), optional
        loss_type: Type of loss
        mask: Optional mask

    Returns:
        Bidirectional warp loss
    """
    # Forward warp loss
    forward_loss = compute_rgb_warp_loss(
        pred_rgb, target_rgb, forward_flow,
        loss_type=loss_type, mask=mask
    )

    if backward_flow is None:
        return forward_loss

    # Backward warp loss: warp rgb_{t+1} to t using backward flow
    B, C, T, H, W = pred_rgb.shape
    T_flow = backward_flow.shape[2]
    num_pairs = min(T - 1, T_flow)

    if num_pairs <= 0:
        return forward_loss

    backward_loss = 0.0
    for t in range(num_pairs):
        rgb_t1 = pred_rgb[:, :, t + 1, :, :]  # (B, 3, H, W)
        bflow_t = backward_flow[:, :2, t, :, :]  # (B, 2, H, W)
        rgb_t_target = target_rgb[:, :, t, :, :]  # (B, 3, H, W)

        rgb_t_warped = warp_image(rgb_t1, bflow_t)

        if loss_type == "l1":
            frame_loss = F.l1_loss(rgb_t_warped, rgb_t_target)
        elif loss_type == "mse":
            frame_loss = F.mse_loss(rgb_t_warped, rgb_t_target)
        else:
            frame_loss = F.l1_loss(rgb_t_warped, rgb_t_target)

        backward_loss = backward_loss + frame_loss

    backward_loss = backward_loss / num_pairs

    return (forward_loss + backward_loss) / 2.0


def compute_occlusion_mask(
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Compute occlusion mask using forward-backward flow consistency.

    Occluded regions are detected where forward and backward flow are inconsistent.

    Args:
        forward_flow: Forward flow (B, 2, H, W)
        backward_flow: Backward flow (B, 2, H, W)
        alpha: Relative threshold
        beta: Absolute threshold

    Returns:
        Occlusion mask (B, 1, H, W) where 1 = valid, 0 = occluded
    """
    # Warp backward flow using forward flow
    warped_backward = warp_image(backward_flow, forward_flow)

    # Check consistency: forward + warped_backward should be ~0
    flow_diff = forward_flow + warped_backward
    flow_diff_sq = (flow_diff ** 2).sum(dim=1, keepdim=True)

    flow_sq = (forward_flow ** 2).sum(dim=1, keepdim=True) + \
              (warped_backward ** 2).sum(dim=1, keepdim=True)

    # Occlusion check
    threshold = alpha * flow_sq + beta
    valid_mask = (flow_diff_sq < threshold).float()

    return valid_mask
