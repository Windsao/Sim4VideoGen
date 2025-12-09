#!/usr/bin/env python3
"""
Script to visualize dataset with RGB, motion vectors, and depth ground truth.
Generates a side-by-side video with RGB | Motion | Depth visualization.
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm


def flow_to_color(flow, max_flow=None):
    """
    Convert optical flow to RGB color image using HSV color wheel.

    Args:
        flow: (H, W, 2) array with dx, dy motion vectors
        max_flow: Maximum flow magnitude for normalization (auto-computed if None)

    Returns:
        (H, W, 3) uint8 RGB image
    """
    h, w = flow.shape[:2]

    # Get dx, dy
    dx = flow[..., 0]
    dy = flow[..., 1]

    # Compute magnitude and angle
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Normalize magnitude
    if max_flow is None:
        max_flow = magnitude.max() + 1e-8
    magnitude = np.clip(magnitude / max_flow, 0, 1)

    # Convert to HSV
    # Hue: direction (angle)
    # Saturation: 1.0
    # Value: magnitude
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # 0-180 for OpenCV
    saturation = np.ones_like(hue) * 255
    value = (magnitude * 255).astype(np.uint8)

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def depth_to_color(depth, colormap=cv2.COLORMAP_VIRIDIS):
    """
    Convert depth map to colorized image.

    Args:
        depth: (H, W) array with depth values
        colormap: OpenCV colormap to use

    Returns:
        (H, W, 3) uint8 RGB image
    """
    # Normalize to 0-255
    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max > depth_min:
        depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(depth_norm, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return colored


def add_text_overlay(img, text, position=(10, 30), font_scale=1.0, color=(255, 255, 255)):
    """Add text overlay to image."""
    img = img.copy()
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    # Add black outline for readability
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return img


def visualize_dataset(
    data_path,
    output_path,
    rgb_subdir="rgb",
    motion_subdir="motion_vectors",
    depth_subdir="distance_to_camera",
    rgb_pattern="rgb_*.png",
    motion_pattern="motion_vectors_*.npy",
    depth_pattern="distance_to_camera_*.npy",
    fps=24,
    max_frames=None,
    layout="horizontal",  # "horizontal", "vertical", or "grid"
    show_labels=True,
    depth_colormap="viridis",
):
    """
    Create visualization video from dataset.

    Args:
        data_path: Path to dataset directory containing rgb/, motion_vectors/, distance_to_camera/
        output_path: Output video file path
        rgb_subdir: Subdirectory name for RGB images
        motion_subdir: Subdirectory name for motion vectors
        depth_subdir: Subdirectory name for depth maps
        rgb_pattern: Glob pattern for RGB images
        motion_pattern: Glob pattern for motion vectors
        depth_pattern: Glob pattern for depth maps
        fps: Frames per second for output video
        max_frames: Maximum number of frames to process (None for all)
        layout: Layout arrangement ("horizontal", "vertical", "grid")
        show_labels: Whether to show text labels on frames
        depth_colormap: Colormap name for depth visualization
    """
    # Setup paths
    rgb_dir = os.path.join(data_path, rgb_subdir)
    motion_dir = os.path.join(data_path, motion_subdir)
    depth_dir = os.path.join(data_path, depth_subdir)

    # Find files
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, rgb_pattern)))
    motion_files = sorted(glob.glob(os.path.join(motion_dir, motion_pattern)))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, depth_pattern)))

    print(f"Found {len(rgb_files)} RGB frames")
    print(f"Found {len(motion_files)} motion vector files")
    print(f"Found {len(depth_files)} depth map files")

    if not rgb_files:
        raise ValueError(f"No RGB images found in {rgb_dir}")

    # Limit frames if specified
    num_frames = len(rgb_files)
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    # Get colormap
    colormap_dict = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }
    cmap = colormap_dict.get(depth_colormap, cv2.COLORMAP_VIRIDIS)

    # Pre-compute max flow for consistent visualization
    print("Computing motion vector statistics...")
    max_flow = 0
    for mf in tqdm(motion_files[:num_frames], desc="Scanning motion"):
        mv = np.load(mf)
        if mv.size > 0:
            flow = mv[..., :2]
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).max()
            max_flow = max(max_flow, mag)
    max_flow = max(max_flow, 1e-8)
    print(f"Max flow magnitude: {max_flow:.3f}")

    # Process first frame to get dimensions
    rgb = np.array(Image.open(rgb_files[0]).convert("RGB"))
    h, w = rgb.shape[:2]

    # Calculate output dimensions based on layout
    if layout == "horizontal":
        out_w, out_h = w * 3, h
    elif layout == "vertical":
        out_w, out_h = w, h * 3
    else:  # grid (2x2 with RGB larger)
        out_w, out_h = w * 2, h * 2

    # Setup video writer
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    print(f"Creating video: {output_path}")
    print(f"Output size: {out_w}x{out_h}, FPS: {fps}")

    # Process frames
    for i in tqdm(range(num_frames), desc="Processing frames"):
        # Load RGB
        rgb = np.array(Image.open(rgb_files[i]).convert("RGB"))

        # Load and visualize motion vectors
        if i < len(motion_files):
            mv = np.load(motion_files[i])
            if mv.size > 0 and not np.all(mv == -1):
                flow = mv[..., :2]
                motion_vis = flow_to_color(flow, max_flow=max_flow)
            else:
                motion_vis = np.zeros_like(rgb)
        else:
            motion_vis = np.zeros_like(rgb)

        # Load and visualize depth
        if i < len(depth_files):
            depth = np.load(depth_files[i])
            if depth.size > 0:
                depth_vis = depth_to_color(depth, colormap=cmap)
            else:
                depth_vis = np.zeros_like(rgb)
        else:
            depth_vis = np.zeros_like(rgb)

        # Resize if needed (motion/depth might have different size)
        if motion_vis.shape[:2] != (h, w):
            motion_vis = cv2.resize(motion_vis, (w, h))
        if depth_vis.shape[:2] != (h, w):
            depth_vis = cv2.resize(depth_vis, (w, h))

        # Add labels
        if show_labels:
            rgb = add_text_overlay(rgb, f"RGB (Frame {i+1})")
            motion_vis = add_text_overlay(motion_vis, "Motion Vectors")
            depth_vis = add_text_overlay(depth_vis, "Depth")

        # Combine based on layout
        if layout == "horizontal":
            combined = np.concatenate([rgb, motion_vis, depth_vis], axis=1)
        elif layout == "vertical":
            combined = np.concatenate([rgb, motion_vis, depth_vis], axis=0)
        else:  # grid
            # RGB top-left (2x size), motion top-right, depth bottom-right
            rgb_large = cv2.resize(rgb, (w, h))
            top_row = np.concatenate([rgb, motion_vis], axis=1)
            # For bottom row, put motion magnitude and depth
            blank = np.zeros_like(rgb)
            if show_labels:
                blank = add_text_overlay(blank, "")
            bottom_row = np.concatenate([blank, depth_vis], axis=1)
            combined = np.concatenate([top_row, bottom_row], axis=0)

        # Write frame (convert RGB to BGR for OpenCV)
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Video saved to: {output_path}")


def visualize_single_frame(
    data_path,
    frame_idx=0,
    output_path=None,
    rgb_subdir="rgb",
    motion_subdir="motion_vectors",
    depth_subdir="distance_to_camera",
    rgb_pattern="rgb_*.png",
    motion_pattern="motion_vectors_*.npy",
    depth_pattern="distance_to_camera_*.npy",
    depth_colormap="viridis",
):
    """
    Visualize a single frame and save/display it.
    """
    import matplotlib.pyplot as plt

    # Setup paths
    rgb_dir = os.path.join(data_path, rgb_subdir)
    motion_dir = os.path.join(data_path, motion_subdir)
    depth_dir = os.path.join(data_path, depth_subdir)

    # Find files
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, rgb_pattern)))
    motion_files = sorted(glob.glob(os.path.join(motion_dir, motion_pattern)))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, depth_pattern)))

    # Get colormap
    colormap_dict = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }
    cmap = colormap_dict.get(depth_colormap, cv2.COLORMAP_VIRIDIS)

    # Load RGB
    rgb = np.array(Image.open(rgb_files[frame_idx]).convert("RGB"))

    # Load motion
    mv = np.load(motion_files[frame_idx])
    if mv.size > 0 and not np.all(mv == -1):
        flow = mv[..., :2]
        motion_vis = flow_to_color(flow)
    else:
        motion_vis = np.zeros_like(rgb)

    # Load depth
    depth = np.load(depth_files[frame_idx])
    depth_vis = depth_to_color(depth, colormap=cmap)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB (Frame {frame_idx})")
    axes[0].axis("off")

    axes[1].imshow(motion_vis)
    axes[1].set_title("Motion Vectors")
    axes[1].axis("off")

    axes[2].imshow(depth_vis)
    axes[2].set_title("Depth")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset with RGB, motion, and depth")
    parser.add_argument("data_path", type=str, help="Path to dataset directory")
    parser.add_argument("--output", "-o", type=str, default="visualization.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--layout", choices=["horizontal", "vertical", "grid"], default="horizontal",
                        help="Layout arrangement")
    parser.add_argument("--no-labels", action="store_true", help="Hide text labels")
    parser.add_argument("--colormap", type=str, default="viridis",
                        choices=["viridis", "plasma", "inferno", "magma", "jet", "turbo"],
                        help="Colormap for depth visualization")
    parser.add_argument("--rgb-subdir", type=str, default="rgb", help="RGB subdirectory name")
    parser.add_argument("--motion-subdir", type=str, default="motion_vectors", help="Motion subdirectory name")
    parser.add_argument("--depth-subdir", type=str, default="distance_to_camera", help="Depth subdirectory name")
    parser.add_argument("--single-frame", type=int, default=None, help="Visualize single frame (index)")

    args = parser.parse_args()

    if args.single_frame is not None:
        # Single frame visualization
        out_path = args.output.replace(".mp4", ".png") if args.output.endswith(".mp4") else args.output
        visualize_single_frame(
            args.data_path,
            frame_idx=args.single_frame,
            output_path=out_path,
            rgb_subdir=args.rgb_subdir,
            motion_subdir=args.motion_subdir,
            depth_subdir=args.depth_subdir,
            depth_colormap=args.colormap,
        )
    else:
        # Video visualization
        visualize_dataset(
            args.data_path,
            args.output,
            rgb_subdir=args.rgb_subdir,
            motion_subdir=args.motion_subdir,
            depth_subdir=args.depth_subdir,
            fps=args.fps,
            max_frames=args.max_frames,
            layout=args.layout,
            show_labels=not args.no_labels,
            depth_colormap=args.colormap,
        )


if __name__ == "__main__":
    main()
