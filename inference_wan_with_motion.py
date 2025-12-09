#!/usr/bin/env python3
"""
Inference script for WAN model with motion vector output.

This script generates videos along with predicted motion vectors,
which can be used for physics-aware video generation and analysis.

Usage:
    python inference_wan_with_motion.py \
        --lora_checkpoint /path/to/lora.safetensors \
        --motion_head_checkpoint /path/to/motion_head.pt \
        --prompt "A physics simulation of objects falling" \
        --output video_output.mp4 \
        --output_motion motion_vectors.npy
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
from diffsynth.models.wan_video_dit_motion import MotionVectorHead
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


class MotionAwareWanPipeline:
    """
    Pipeline wrapper that adds motion vector prediction capability.
    """

    def __init__(
        self,
        pipe: WanVideoPipeline,
        motion_head: MotionVectorHead,
        motion_channels: int = 4,
    ):
        self.pipe = pipe
        self.motion_head = motion_head
        self.motion_channels = motion_channels

        # Move motion head to same device as pipeline
        self.motion_head = self.motion_head.to(
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
        motion_at_timestep: Optional[int] = None,
        **kwargs,
    ):
        """
        Generate video with optional motion vector prediction.

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
            motion_at_timestep: Predict motion at specific timestep (default: last step)
            **kwargs: Additional arguments passed to pipeline

        Returns:
            If return_motion is False: video frames
            If return_motion is True: (video frames, motion vectors)
        """
        # Reset captured features
        self._captured_features = None
        self._captured_timestep = None

        # Generate video
        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            tiled=tiled,
            **kwargs,
        )

        if not return_motion:
            return video

        # Predict motion vectors from captured features
        motion_vectors = None
        if self._captured_features is not None:
            # Calculate grid size
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

            motion_vectors = self.predict_motion_vectors(
                self._captured_features,
                timestep,
                grid_size,
            )

            # Convert to numpy for saving
            motion_vectors = motion_vectors.cpu().numpy()

        return video, motion_vectors


def load_motion_head(
    checkpoint_path: str,
    dim: int,
    motion_channels: int = 4,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> MotionVectorHead:
    """
    Load a trained motion head checkpoint.

    Args:
        checkpoint_path: Path to motion head checkpoint
        dim: DiT hidden dimension
        motion_channels: Number of motion channels
        patch_size: DiT patch size
        device: Device to load to
        dtype: Data type

    Returns:
        Loaded MotionVectorHead
    """
    motion_head = MotionVectorHead(
        dim=dim,
        motion_channels=motion_channels,
        patch_size=patch_size,
        eps=1e-6,
        output_scale=1.0,
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        motion_head.load_state_dict(state_dict)
        print(f"Loaded motion head from: {checkpoint_path}")
    else:
        print("Warning: No motion head checkpoint provided, using random initialization")

    return motion_head.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="WAN inference with motion vector output")
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

    args = parser.parse_args()

    print("=" * 60)
    print("WAN Video Generation with Motion Vectors")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Output video: {args.output}")
    print(f"Output motion: {args.output_motion}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames}")
    print()

    # Load base pipeline
    print("Loading WAN pipeline...")
    model_base = args.model_base_path

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
    if args.lora_checkpoint:
        print(f"Loading LoRA from: {args.lora_checkpoint}")
        lora_state_dict = load_state_dict(args.lora_checkpoint)
        lora_loader = GeneralLoRALoader(device=args.device, torch_dtype=torch.bfloat16)
        lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)

    # Enable VRAM management
    pipe.enable_vram_management()

    # Create motion head
    print("Setting up motion head...")
    motion_head = load_motion_head(
        checkpoint_path=args.motion_head_checkpoint,
        dim=pipe.dit.dim,
        motion_channels=args.motion_channels,
        patch_size=pipe.dit.patch_size,
        device=args.device,
        dtype=torch.bfloat16,
    )

    # Create motion-aware pipeline
    motion_pipe = MotionAwareWanPipeline(
        pipe=pipe,
        motion_head=motion_head,
        motion_channels=args.motion_channels,
    )

    # Generate video with motion
    print("Generating video...")
    return_motion = args.output_motion is not None

    result = motion_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        return_motion=return_motion,
    )

    if return_motion:
        video, motion_vectors = result
    else:
        video = result
        motion_vectors = None

    # Save video
    print(f"Saving video to: {args.output}")
    save_video(video, args.output, fps=args.fps, quality=5)

    # Save motion vectors
    if motion_vectors is not None and args.output_motion:
        print(f"Saving motion vectors to: {args.output_motion}")
        np.save(args.output_motion, motion_vectors)
        print(f"Motion vector shape: {motion_vectors.shape}")

    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
