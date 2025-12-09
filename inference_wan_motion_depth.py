#!/usr/bin/env python3
"""
Inference script for WAN model with motion vector and depth output (T2V mode).

This script generates videos along with predicted motion vectors and depth maps,
which can be used for physics-aware video generation and analysis.

Note: This uses Text-to-Video (T2V) mode. Input images are not supported
as the I2V model is not available on this system.

Usage:
    python inference_wan_motion_depth.py \
        --motion_head_checkpoint /path/to/motion_head.safetensors \
        --depth_head_checkpoint /path/to/depth_head.safetensors \
        --prompt "A physics simulation of objects falling" \
        --output video_output.mp4 \
        --output_motion motion_vectors.npy \
        --output_depth depth_maps.npy

    # With LoRA:
    python inference_wan_motion_depth.py \
        --lora_checkpoint /path/to/lora.safetensors \
        --motion_head_checkpoint /path/to/motion_head.safetensors \
        --prompt "A physics simulation of objects falling" \
        --output video_output.mp4
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple
from einops import rearrange

from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.wan_video_dit_motion import MotionVectorHead, DepthHead
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


class MotionDepthWanPipeline:
    """
    Pipeline wrapper that adds motion vector and depth prediction capability.
    """

    def __init__(
        self,
        pipe: WanVideoPipeline,
        motion_head: Optional[MotionVectorHead] = None,
        depth_head: Optional[DepthHead] = None,
        motion_channels: int = 4,
    ):
        self.pipe = pipe
        self.motion_head = motion_head
        self.depth_head = depth_head
        self.motion_channels = motion_channels

        # Move heads to same device as pipeline
        if self.motion_head is not None:
            self.motion_head = self.motion_head.to(
                device=pipe.device,
                dtype=pipe.torch_dtype
            )
        if self.depth_head is not None:
            self.depth_head = self.depth_head.to(
                device=pipe.device,
                dtype=pipe.torch_dtype
            )

        # Store features captured during inference
        self._captured_features = None
        self._captured_timestep = None

        # Setup hook to capture DiT features
        self._setup_feature_capture()

    def _setup_feature_capture(self):
        """Setup hook to capture intermediate DiT features."""
        def capture_hook(module, input, output):
            self._captured_features = input[0].detach()
            if len(input) > 1:
                self._captured_timestep = input[1].detach()

        self.pipe.dit.head.register_forward_hook(capture_hook)

    def predict_motion_vectors(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Predict motion vectors from DiT features.

        Args:
            features: DiT features of shape (B, S, D)
            timestep: Timestep tensor
            grid_size: (F, H, W) grid dimensions

        Returns:
            Motion vectors of shape (B, C, F, H, W)
        """
        # Ensure timestep is 1D
        if timestep.dim() > 1:
            timestep = timestep.flatten()[:1]

        # Get time embedding
        t_embed = self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep)
        )

        # Predict motion
        motion_pred = self.motion_head(features, t_embed)

        # Unpatchify
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size

        motion_pred = rearrange(
            motion_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
            c=self.motion_channels
        )

        return motion_pred

    def predict_depth(
        self,
        features: torch.Tensor,
        timestep: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Predict depth maps from DiT features.

        Args:
            features: DiT features of shape (B, S, D)
            timestep: Timestep tensor
            grid_size: (F, H, W) grid dimensions

        Returns:
            Depth maps of shape (B, 1, F, H, W)
        """
        # Ensure timestep is 1D
        if timestep.dim() > 1:
            timestep = timestep.flatten()[:1]

        # Get time embedding
        t_embed = self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep)
        )

        # Predict depth
        depth_pred = self.depth_head(features, t_embed)

        # Unpatchify
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size

        depth_pred = rearrange(
            depth_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
            c=1  # Single channel depth
        )

        return depth_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        seed: int = 42,
        num_inference_steps: int = 50,
        cfg_scale: float = 7.0,
        tiled: bool = False,
        return_motion: bool = True,
        return_depth: bool = True,
        **kwargs,
    ):
        """
        Generate video with optional motion vector and depth prediction (T2V mode).

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt
            height: Video height
            width: Video width
            num_frames: Number of frames
            seed: Random seed
            num_inference_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            tiled: Use tiled generation for memory efficiency
            return_motion: Whether to return motion vectors
            return_depth: Whether to return depth maps
            **kwargs: Additional arguments passed to pipeline

        Returns:
            Dict with 'video', 'motion_vectors' (optional), 'depth_maps' (optional)
        """
        # Reset captured features
        self._captured_features = None
        self._captured_timestep = None

        # Build pipeline kwargs (T2V mode)
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
        }

        pipe_kwargs.update(kwargs)

        # Generate video
        video = self.pipe(**pipe_kwargs)

        result = {"video": video}

        # Calculate grid size for unpatchify
        patch_size = self.pipe.dit.patch_size
        latent_f = (num_frames - 1) // 4 + 1  # VAE temporal compression
        latent_h = height // 8  # VAE spatial compression
        latent_w = width // 8

        f = latent_f // patch_size[0]
        h = latent_h // patch_size[1]
        w = latent_w // patch_size[2]
        grid_size = (f, h, w)

        # Use last timestep if not specified
        if self._captured_timestep is None:
            timestep = torch.tensor([0.0], device=self.pipe.device, dtype=self.pipe.torch_dtype)
        else:
            timestep = self._captured_timestep

        # Predict motion vectors if requested and head is available
        if return_motion and self.motion_head is not None and self._captured_features is not None:
            motion_vectors = self.predict_motion_vectors(
                self._captured_features,
                timestep,
                grid_size,
            )
            # Convert to float32 before numpy (bfloat16 not supported)
            result["motion_vectors"] = motion_vectors.float().cpu().numpy()

        # Predict depth maps if requested and head is available
        if return_depth and self.depth_head is not None and self._captured_features is not None:
            depth_maps = self.predict_depth(
                self._captured_features,
                timestep,
                grid_size,
            )
            # Convert to float32 before numpy (bfloat16 not supported)
            result["depth_maps"] = depth_maps.float().cpu().numpy()

        return result


def load_motion_head(
    checkpoint_path: str,
    dim: int,
    motion_channels: int = 4,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[MotionVectorHead]:
    """Load a trained motion head checkpoint (supports .safetensors and .pt)."""
    if not checkpoint_path:
        return None

    motion_head = MotionVectorHead(
        dim=dim,
        motion_channels=motion_channels,
        patch_size=patch_size,
        eps=1e-6,
        output_scale=1.0,
    )

    if Path(checkpoint_path).exists():
        # Use load_state_dict which handles both .safetensors and .pt files
        state_dict = load_state_dict(checkpoint_path)

        # Handle prefixed keys from training checkpoint
        # The checkpoint may have "motion_head." or "head." prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            # Try to find motion head keys with various prefixes
            if key.startswith("motion_head."):
                new_key = key[len("motion_head."):]
                new_state_dict[new_key] = value
            elif key.startswith("head.") and "motion" not in key:
                # Skip the main "head" keys (for noise prediction)
                continue
            else:
                new_state_dict[key] = value

        # If we found motion_head prefixed keys, use the new dict
        if any(k.startswith("motion_head.") for k in state_dict.keys()):
            state_dict = new_state_dict
            print(f"  Extracted motion_head keys: {list(state_dict.keys())}")

        # Load with strict=False and check what was loaded
        missing, unexpected = motion_head.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: Missing keys: {missing}")
        if unexpected:
            print(f"  Warning: Unexpected keys: {unexpected}")
        print(f"Loaded motion head from: {checkpoint_path}")
    else:
        print(f"Warning: Motion head checkpoint not found: {checkpoint_path}")
        return None

    return motion_head.to(device=device, dtype=dtype)


def load_depth_head(
    checkpoint_path: str,
    dim: int,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[DepthHead]:
    """Load a trained depth head checkpoint (supports .safetensors and .pt)."""
    if not checkpoint_path:
        return None

    depth_head = DepthHead(
        dim=dim,
        depth_channels=1,
        patch_size=patch_size,
        eps=1e-6,
        output_scale=1.0,
    )

    if Path(checkpoint_path).exists():
        # Use load_state_dict which handles both .safetensors and .pt files
        state_dict = load_state_dict(checkpoint_path)

        # Handle prefixed keys from training checkpoint
        # The checkpoint may have "depth_head." prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("depth_head."):
                new_key = key[len("depth_head."):]
                new_state_dict[new_key] = value
            elif key.startswith("head.") and "depth" not in key:
                # Skip non-depth head keys
                continue
            else:
                new_state_dict[key] = value

        # If we found depth_head prefixed keys, use the new dict
        if any(k.startswith("depth_head.") for k in state_dict.keys()):
            state_dict = new_state_dict

        depth_head.load_state_dict(state_dict, strict=False)
        print(f"Loaded depth head from: {checkpoint_path}")
    else:
        print(f"Warning: Depth head checkpoint not found: {checkpoint_path}")
        return None

    return depth_head.to(device=device, dtype=dtype)


def visualize_motion(motion_vectors: np.ndarray, output_path: str):
    """
    Visualize motion vectors as color-coded flow images.

    Args:
        motion_vectors: Motion vectors of shape (B, C, F, H, W)
        output_path: Path to save visualization
    """
    import cv2
    from PIL import Image

    # Take first batch
    motion = motion_vectors[0]  # (C, F, H, W)

    # Extract dx and dy (first 2 channels)
    dx = motion[0]  # (F, H, W)
    dy = motion[1]  # (F, H, W)

    frames = []
    for i in range(dx.shape[0]):
        # Compute magnitude and angle
        mag = np.sqrt(dx[i]**2 + dy[i]**2)
        ang = np.arctan2(dy[i], dx[i])

        # Convert to HSV color space
        hsv = np.zeros((*dx.shape[1:], 3), dtype=np.uint8)
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue
        hsv[..., 1] = 255  # Saturation
        hsv[..., 2] = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8) if mag.max() > 0 else 0  # Value

        # Convert to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        frames.append(Image.fromarray(rgb))

    # Save as video
    save_video(frames, output_path, fps=15)
    print(f"Saved motion visualization to: {output_path}")


def visualize_depth(depth_maps: np.ndarray, output_path: str):
    """
    Visualize depth maps as grayscale video.

    Args:
        depth_maps: Depth maps of shape (B, 1, F, H, W)
        output_path: Path to save visualization
    """
    from PIL import Image

    # Take first batch
    depth = depth_maps[0, 0]  # (F, H, W)

    # Normalize to [0, 255]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth, dtype=np.uint8)

    frames = []
    for i in range(depth_normalized.shape[0]):
        # Convert to RGB (grayscale)
        frame = np.stack([depth_normalized[i]] * 3, axis=-1)
        frames.append(Image.fromarray(frame))

    # Save as video
    save_video(frames, output_path, fps=15)
    print(f"Saved depth visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WAN inference with motion vector and depth output")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--motion_head_checkpoint",
        type=str,
        default=None,
        help="Path to motion head checkpoint",
    )
    parser.add_argument(
        "--depth_head_checkpoint",
        type=str,
        default=None,
        help="Path to depth head checkpoint",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to input image for I2V generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A physics simulation of objects interacting",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative text prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_video.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--output_motion",
        type=str,
        default=None,
        help="Output path for motion vectors (.npy)",
    )
    parser.add_argument(
        "--output_depth",
        type=str,
        default=None,
        help="Output path for depth maps (.npy)",
    )
    parser.add_argument(
        "--output_motion_vis",
        type=str,
        default=None,
        help="Output path for motion visualization video",
    )
    parser.add_argument(
        "--output_depth_vis",
        type=str,
        default=None,
        help="Output path for depth visualization video",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--model_base_path",
        type=str,
        default="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI",
        help="Base path for WAN models",
    )
    parser.add_argument(
        "--motion_channels",
        type=int,
        default=4,
        help="Number of motion vector channels",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Output video FPS",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.0,
        help="CFG scale",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WAN Video Generation with Motion Vectors and Depth")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Input image: {args.input_image}")
    print(f"Output video: {args.output}")
    print(f"Output motion: {args.output_motion}")
    print(f"Output depth: {args.output_depth}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames}")
    print()

    # Note: input_image is kept for compatibility but not used in T2V mode
    if args.input_image:
        print(f"Note: input_image is provided but will be ignored (T2V mode only)")
        print(f"  Input image: {args.input_image}")

    # Load base pipeline (T2V model - I2V model is not available)
    print("Loading WAN pipeline (T2V mode)...")
    model_base = args.model_base_path

    # Text-to-video model
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(
                path=f"{model_base}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(
                path=f"{model_base}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu"
            ),
            ModelConfig(
                path=f"{model_base}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                offload_device="cpu"
            ),
        ],
        tokenizer_config=ModelConfig(
            path=f"{model_base}/Wan2.1-T2V-1.3B/google/umt5-xxl"
        ),
    )

    # Load LoRA if provided
    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        print(f"Loading LoRA from: {args.lora_checkpoint}")
        lora_state_dict = load_state_dict(args.lora_checkpoint)
        lora_loader = GeneralLoRALoader(device=args.device, torch_dtype=torch.bfloat16)
        lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)

    # Enable VRAM management
    pipe.enable_vram_management()

    # Load motion head if provided
    motion_head = load_motion_head(
        checkpoint_path=args.motion_head_checkpoint,
        dim=pipe.dit.dim,
        motion_channels=args.motion_channels,
        patch_size=pipe.dit.patch_size,
        device=args.device,
        dtype=torch.bfloat16,
    )

    # Load depth head if provided
    depth_head = load_depth_head(
        checkpoint_path=args.depth_head_checkpoint,
        dim=pipe.dit.dim,
        patch_size=pipe.dit.patch_size,
        device=args.device,
        dtype=torch.bfloat16,
    )

    # Create motion-depth pipeline
    motion_depth_pipe = MotionDepthWanPipeline(
        pipe=pipe,
        motion_head=motion_head,
        depth_head=depth_head,
        motion_channels=args.motion_channels,
    )

    # Generate video with motion and depth
    print("Generating video...")
    return_motion = args.output_motion is not None or args.output_motion_vis is not None
    return_depth = args.output_depth is not None or args.output_depth_vis is not None

    result = motion_depth_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        return_motion=return_motion,
        return_depth=return_depth,
    )

    video = result["video"]
    motion_vectors = result.get("motion_vectors")
    depth_maps = result.get("depth_maps")

    # Save video
    print(f"Saving video to: {args.output}")
    save_video(video, args.output, fps=args.fps, quality=5)

    # Save motion vectors
    if motion_vectors is not None:
        if args.output_motion:
            print(f"Saving motion vectors to: {args.output_motion}")
            np.save(args.output_motion, motion_vectors)
            print(f"Motion vector shape: {motion_vectors.shape}")

        if args.output_motion_vis:
            print(f"Creating motion visualization...")
            visualize_motion(motion_vectors, args.output_motion_vis)

    # Save depth maps
    if depth_maps is not None:
        if args.output_depth:
            print(f"Saving depth maps to: {args.output_depth}")
            np.save(args.output_depth, depth_maps)
            print(f"Depth map shape: {depth_maps.shape}")

        if args.output_depth_vis:
            print(f"Creating depth visualization...")
            visualize_depth(depth_maps, args.output_depth_vis)

    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
