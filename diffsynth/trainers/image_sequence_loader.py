"""
Custom data loader operators for image sequence datasets.
This module provides operators to load video data from image sequences,
with optional support for loading motion vectors.
"""

import os
import glob
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
from .unified_dataset import DataProcessingOperator


class LoadImageSequence(DataProcessingOperator):
    """
    Load a sequence of images from a directory as video frames.

    This operator expects a path to a directory containing sequentially numbered images
    (e.g., rgb_0001.png, rgb_0002.png, ...) and loads them as a video sequence.

    Args:
        num_frames: Target number of frames to load (default: 81)
        time_division_factor: Frame count must be divisible by this (default: 4)
        time_division_remainder: Required remainder when dividing by time_division_factor (default: 1)
        frame_processor: Optional processing function to apply to each frame
        pattern: Glob pattern to match image files (default: "*.png")
    """

    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        pattern="*.png"
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.pattern = pattern

    def get_num_frames(self, total_frames):
        """Calculate the number of frames to load based on constraints."""
        num_frames = min(self.num_frames, total_frames)
        # Ensure num_frames satisfies: num_frames % time_division_factor == time_division_remainder
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        """
        Load image sequence from directory.

        Args:
            data: Path to directory containing image sequence

        Returns:
            List of PIL Image objects
        """
        if not os.path.isdir(data):
            raise ValueError(f"Expected directory path, got: {data}")

        # Find all images matching the pattern
        image_paths = sorted(glob.glob(os.path.join(data, self.pattern)))

        if not image_paths:
            raise ValueError(f"No images found matching pattern '{self.pattern}' in {data}")

        total_frames = len(image_paths)
        num_frames = self.get_num_frames(total_frames)

        # Load frames evenly distributed across the sequence
        frames = []
        if num_frames == total_frames:
            # Use all frames
            selected_indices = list(range(total_frames))
        else:
            # Sample frames evenly
            selected_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        for idx in selected_indices:
            frame = Image.open(image_paths[idx]).convert("RGB")
            frame = self.frame_processor(frame)
            frames.append(frame)

        return frames


class LoadImageSequenceFromList(DataProcessingOperator):
    """
    Load image sequences when given a list of image paths.

    This is useful when you have a metadata CSV that lists individual image paths
    rather than directory paths.

    Args:
        num_frames: Target number of frames to load (default: 81)
        time_division_factor: Frame count must be divisible by this (default: 4)
        time_division_remainder: Required remainder when dividing by time_division_factor (default: 1)
        frame_processor: Optional processing function to apply to each frame
    """

    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def get_num_frames(self, total_frames):
        """Calculate the number of frames to load based on constraints."""
        num_frames = min(self.num_frames, total_frames)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def __call__(self, data: List[str]):
        """
        Load image sequence from list of paths.

        Args:
            data: List of paths to image files

        Returns:
            List of PIL Image objects
        """
        if not isinstance(data, list):
            raise ValueError(f"Expected list of image paths, got: {type(data)}")

        total_frames = len(data)
        num_frames = self.get_num_frames(total_frames)

        # Load frames evenly distributed across the sequence
        frames = []
        if num_frames == total_frames:
            selected_indices = list(range(total_frames))
        else:
            selected_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        for idx in selected_indices:
            frame = Image.open(data[idx]).convert("RGB")
            frame = self.frame_processor(frame)
            frames.append(frame)

        return frames


class LoadImageSequenceWithMotion(DataProcessingOperator):
    """
    Load a sequence of images with corresponding motion vectors and depth maps from a directory.

    This operator expects a directory structure like:
        base_dir/
            rgb/
                rgb_0001.png
                rgb_0002.png
                ...
            motion_vectors/
                motion_vectors_0001.npy
                motion_vectors_0002.npy
                ...
            distance_to_camera/  (optional)
                distance_to_camera_0001.npy
                distance_to_camera_0002.npy
                ...

    Motion vectors are numpy arrays with shape (H, W, 4) containing:
    - Channel 0: dx (horizontal motion)
    - Channel 1: dy (vertical motion)
    - Channel 2-3: additional motion information

    Depth maps are numpy arrays with shape (H, W) containing distance to camera.

    Args:
        num_frames: Target number of frames to load (default: 81)
        time_division_factor: Frame count must be divisible by this (default: 4)
        time_division_remainder: Required remainder when dividing by time_division_factor (default: 1)
        frame_processor: Optional processing function to apply to each frame
        image_pattern: Glob pattern to match image files (default: "*.png")
        motion_pattern: Glob pattern to match motion vector files (default: "*.npy")
        depth_pattern: Glob pattern to match depth files (default: "*.npy")
        rgb_subdir: Subdirectory name for RGB images (default: "rgb")
        motion_subdir: Subdirectory name for motion vectors (default: "motion_vectors")
        depth_subdir: Subdirectory name for depth maps (default: "distance_to_camera")
        motion_channels: Number of motion vector channels to use (default: 4)
        normalize_motion: Whether to normalize motion vectors (default: False)
        motion_scale: Scale factor for motion vectors (default: 1.0)
        depth_scale: Scale factor for depth values (default: 1.0)
        normalize_depth: Whether to normalize depth to [0, 1] range (default: False)
        load_depth: Whether to load depth data (default: True)
    """

    def __init__(
        self,
        num_frames: int = 81,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        frame_processor=lambda x: x,
        image_pattern: str = "*.png",
        motion_pattern: str = "*.npy",
        depth_pattern: str = "*.npy",
        rgb_subdir: str = "rgb",
        motion_subdir: str = "motion_vectors",
        depth_subdir: str = "distance_to_camera",
        motion_channels: int = 4,
        normalize_motion: bool = False,
        motion_scale: float = 1.0,
        depth_scale: float = 1.0,
        normalize_depth: bool = False,
        load_depth: bool = True,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.image_pattern = image_pattern
        self.motion_pattern = motion_pattern
        self.depth_pattern = depth_pattern
        self.rgb_subdir = rgb_subdir
        self.motion_subdir = motion_subdir
        self.depth_subdir = depth_subdir
        self.motion_channels = motion_channels
        self.normalize_motion = normalize_motion
        self.motion_scale = motion_scale
        self.depth_scale = depth_scale
        self.normalize_depth = normalize_depth
        self.load_depth = load_depth

    def get_num_frames(self, total_frames: int) -> int:
        """Calculate the number of frames to load based on constraints."""
        num_frames = min(self.num_frames, total_frames)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def load_motion_vector(self, path: str, target_size: tuple = None) -> np.ndarray:
        """
        Load and preprocess a single motion vector file.

        Args:
            path: Path to motion vector file
            target_size: Optional (height, width) to resize motion vectors to match frames
        """
        mv = np.load(path).astype(np.float32)

        # Handle empty motion vectors (e.g., first frame)
        if mv.size == 0:
            return None

        # Select channels
        if mv.shape[-1] > self.motion_channels:
            mv = mv[..., :self.motion_channels]

        # Resize to target size if specified
        if target_size is not None and mv.shape[:2] != target_size:
            import cv2
            # Resize each channel separately
            resized_channels = []
            for c in range(mv.shape[-1]):
                resized = cv2.resize(mv[..., c], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                resized_channels.append(resized)
            mv = np.stack(resized_channels, axis=-1)

        # Apply scaling
        mv = mv * self.motion_scale

        # Optional normalization
        if self.normalize_motion:
            # Normalize to [-1, 1] based on image dimensions
            h, w = mv.shape[:2]
            mv[..., 0] = mv[..., 0] / w  # dx normalized by width
            mv[..., 1] = mv[..., 1] / h  # dy normalized by height

        return mv

    def load_depth_map(self, path: str, target_size: tuple = None) -> np.ndarray:
        """
        Load and preprocess a single depth map file.

        Args:
            path: Path to depth map file
            target_size: Optional (height, width) to resize depth map to match frames
        """
        depth = np.load(path).astype(np.float32)

        # Handle empty depth maps
        if depth.size == 0:
            return None

        # Handle NaN and Inf values
        if np.isnan(depth).any() or np.isinf(depth).any():
            # Replace NaN with 0 and Inf with large finite values
            depth = np.nan_to_num(depth, nan=0.0, posinf=1e6, neginf=0.0)

        # Apply scaling
        depth = depth * self.depth_scale

        # Optional normalization to [0, 1]
        if self.normalize_depth:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)

        # Resize to target size if specified
        if target_size is not None and depth.shape != target_size:
            import cv2
            depth = cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        return depth

    def __call__(self, data: str) -> Dict[str, Any]:
        """
        Load image sequence, motion vectors, and depth maps from directory.

        Args:
            data: Path to base directory containing rgb/, motion_vectors/, and distance_to_camera/ subdirs

        Returns:
            Dictionary with:
                - 'video': List of PIL Image objects
                - 'motion_vectors': torch.Tensor of shape (C, F-1, H, W) or None
                - 'depth_maps': torch.Tensor of shape (1, F, H, W) or None
                - 'selected_indices': List of frame indices used
        """
        if not os.path.isdir(data):
            raise ValueError(f"Expected directory path, got: {data}")

        # Find RGB images
        rgb_dir = os.path.join(data, self.rgb_subdir)
        if not os.path.isdir(rgb_dir):
            # Fallback: maybe the path is directly to the rgb directory
            rgb_dir = data

        image_paths = sorted(glob.glob(os.path.join(rgb_dir, self.image_pattern)))
        if not image_paths:
            raise ValueError(f"No images found matching pattern '{self.image_pattern}' in {rgb_dir}")

        total_frames = len(image_paths)
        num_frames = self.get_num_frames(total_frames)

        # Determine selected frame indices
        if num_frames == total_frames:
            selected_indices = list(range(total_frames))
        else:
            selected_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        # Load frames
        frames = []
        for idx in selected_indices:
            frame = Image.open(image_paths[idx]).convert("RGB")
            frame = self.frame_processor(frame)
            frames.append(frame)

        # Find and load motion vectors
        motion_dir = os.path.join(data, self.motion_subdir)
        motion_vectors = None

        # Get target size from processed frames
        target_size = (frames[0].height, frames[0].width) if frames else None

        if os.path.isdir(motion_dir):
            motion_paths = sorted(glob.glob(os.path.join(motion_dir, self.motion_pattern)))

            if motion_paths:
                # Motion vectors represent inter-frame motion, so we need F-1 vectors for F frames
                # motion_vectors[i] represents motion from frame i to frame i+1
                motion_list = []

                for i, idx in enumerate(selected_indices[:-1]):
                    # For motion from frame idx to frame idx+1
                    # We need motion_vectors file at index idx+1 (1-indexed in filenames)
                    # or we approximate with the nearest available
                    next_idx = selected_indices[i + 1]

                    # Motion vector files are typically 1-indexed and represent motion TO that frame
                    # So motion_vectors_0002.npy contains motion from frame 1 to frame 2
                    mv_idx = min(next_idx, len(motion_paths) - 1)

                    if mv_idx < len(motion_paths):
                        mv = self.load_motion_vector(motion_paths[mv_idx], target_size=target_size)
                        if mv is not None:
                            motion_list.append(mv)
                        else:
                            # Create zero motion for empty/invalid frames
                            if motion_list:
                                mv = np.zeros_like(motion_list[-1])
                            elif target_size:
                                mv = np.zeros((target_size[0], target_size[1], self.motion_channels), dtype=np.float32)
                            else:
                                # Fallback shape
                                mv = np.zeros((frames[0].height, frames[0].width, self.motion_channels), dtype=np.float32)
                            motion_list.append(mv)

                if motion_list:
                    # Stack to (F-1, H, W, C) and convert to torch tensor
                    motion_vectors = torch.from_numpy(np.stack(motion_list, axis=0))
                    # Rearrange to (C, F-1, H, W) for consistency with video format
                    motion_vectors = motion_vectors.permute(3, 0, 1, 2)

        # Find and load depth maps
        depth_maps = None

        if self.load_depth:
            depth_dir = os.path.join(data, self.depth_subdir)

            if os.path.isdir(depth_dir):
                depth_paths = sorted(glob.glob(os.path.join(depth_dir, self.depth_pattern)))

                if depth_paths:
                    depth_list = []
                    # Get target size from processed frames
                    target_size = (frames[0].height, frames[0].width) if frames else None

                    # First pass: load all valid depths
                    temp_depths = []
                    for idx in selected_indices:
                        # Depth maps are per-frame, aligned with RGB frames
                        depth_idx = min(idx, len(depth_paths) - 1)

                        if depth_idx < len(depth_paths):
                            depth = self.load_depth_map(depth_paths[depth_idx], target_size=target_size)
                            temp_depths.append(depth)  # Can be None for empty files
                        else:
                            temp_depths.append(None)

                    # Second pass: fill in None values with nearest valid depth
                    for i, depth in enumerate(temp_depths):
                        if depth is not None:
                            depth_list.append(depth)
                        else:
                            # Find nearest valid depth (prefer forward, then backward)
                            found = False
                            for j in range(1, len(temp_depths)):
                                # Check forward
                                if i + j < len(temp_depths) and temp_depths[i + j] is not None:
                                    depth_list.append(temp_depths[i + j].copy())
                                    found = True
                                    break
                                # Check backward
                                if i - j >= 0 and temp_depths[i - j] is not None:
                                    depth_list.append(temp_depths[i - j].copy())
                                    found = True
                                    break
                            if not found:
                                # No valid depth found anywhere, create zeros
                                if target_size:
                                    depth_list.append(np.zeros(target_size, dtype=np.float32))
                                else:
                                    depth_list.append(np.zeros((frames[0].height, frames[0].width), dtype=np.float32))

                    if depth_list:
                        # Stack to (F, H, W) and convert to torch tensor
                        depth_maps = torch.from_numpy(np.stack(depth_list, axis=0))
                        # Add channel dimension: (F, H, W) -> (1, F, H, W)
                        depth_maps = depth_maps.unsqueeze(0)

        return {
            'video': frames,
            'motion_vectors': motion_vectors,
            'depth_maps': depth_maps,
            'selected_indices': selected_indices,
        }


class LoadMotionVectors(DataProcessingOperator):
    """
    Standalone operator to load motion vectors from a directory.

    Use this when you want to load motion vectors separately from images.

    Args:
        num_frames: Target number of frames (motion vectors will be F-1)
        time_division_factor: Frame count must be divisible by this
        time_division_remainder: Required remainder
        pattern: Glob pattern to match motion vector files
        motion_channels: Number of channels to use
        normalize: Whether to normalize motion vectors
        scale: Scale factor
    """

    def __init__(
        self,
        num_frames: int = 81,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        pattern: str = "*.npy",
        motion_channels: int = 4,
        normalize: bool = False,
        scale: float = 1.0,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.pattern = pattern
        self.motion_channels = motion_channels
        self.normalize = normalize
        self.scale = scale

    def get_num_frames(self, total_frames: int) -> int:
        num_frames = min(self.num_frames, total_frames)
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def __call__(self, data: str) -> torch.Tensor:
        """
        Load motion vectors from directory.

        Args:
            data: Path to directory containing motion vector .npy files

        Returns:
            torch.Tensor of shape (C, F-1, H, W)
        """
        if not os.path.isdir(data):
            raise ValueError(f"Expected directory path, got: {data}")

        motion_paths = sorted(glob.glob(os.path.join(data, self.pattern)))
        if not motion_paths:
            raise ValueError(f"No motion vectors found matching '{self.pattern}' in {data}")

        total_frames = len(motion_paths)
        num_frames = self.get_num_frames(total_frames)

        if num_frames == total_frames:
            selected_indices = list(range(total_frames))
        else:
            selected_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        motion_list = []
        for idx in selected_indices[:-1]:  # F-1 motion vectors for F frames
            next_idx = selected_indices[selected_indices.index(idx) + 1] if idx in selected_indices else idx + 1
            mv_idx = min(next_idx, len(motion_paths) - 1)

            mv = np.load(motion_paths[mv_idx]).astype(np.float32)
            if mv.size == 0:
                if motion_list:
                    mv = np.zeros_like(motion_list[-1])
                else:
                    continue
            else:
                if mv.shape[-1] > self.motion_channels:
                    mv = mv[..., :self.motion_channels]
                mv = mv * self.scale
                if self.normalize:
                    h, w = mv.shape[:2]
                    mv[..., 0] = mv[..., 0] / w
                    mv[..., 1] = mv[..., 1] / h

            motion_list.append(mv)

        if not motion_list:
            raise ValueError(f"No valid motion vectors found in {data}")

        motion_vectors = torch.from_numpy(np.stack(motion_list, axis=0))
        motion_vectors = motion_vectors.permute(3, 0, 1, 2)  # (F-1, H, W, C) -> (C, F-1, H, W)

        return motion_vectors
