#!/usr/bin/env python3
"""
Inference script for WAN2.2-TI2V Stage 2 LoRA checkpoints.

Loads the WAN2.2-TI2V-5B base model and applies Stage 2 LoRA weights,
optionally loading motion/depth heads for evaluation.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffsynth import save_video, load_state_dict
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.models.wan_video_dit_motion import MotionVectorHead, DepthHead
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


class MotionDepthWanPipeline:
    def __init__(
        self,
        pipe: WanVideoPipeline,
        motion_head: Optional[MotionVectorHead],
        depth_head: Optional[DepthHead],
        motion_channels: int = 4,
    ):
        self.pipe = pipe
        self.motion_head = motion_head
        self.depth_head = depth_head
        self.motion_channels = motion_channels
        self._captured_features = None
        self._captured_timestep = None
        self._setup_feature_capture()

        if self.motion_head is not None:
            self.motion_head = self.motion_head.to(device=pipe.device, dtype=pipe.torch_dtype)
        if self.depth_head is not None:
            self.depth_head = self.depth_head.to(device=pipe.device, dtype=pipe.torch_dtype)

    def _setup_feature_capture(self):
        def capture_hook(module, input, output):
            self._captured_features = input[0].detach()
            if len(input) > 1:
                self._captured_timestep = input[1].detach()

        self.pipe.dit.head.register_forward_hook(capture_hook)

    def _grid_size(self, num_frames: int, height: int, width: int, seq_len: int) -> Tuple[int, int, int]:
        patch_size = self.pipe.dit.patch_size
        f = (num_frames - 1) // 4 + 1
        if f <= 0:
            raise ValueError(f"Invalid num_frames for grid size: {num_frames}")
        if seq_len % f != 0:
            raise ValueError(f"Sequence length {seq_len} not divisible by f={f}")
        hw = seq_len // f
        h_guess = int(round(hw ** 0.5))
        if h_guess * h_guess == hw:
            h = h_guess
            w = h_guess
        else:
            latent_h = height // 8
            latent_w = width // 8
            base_h = max(1, latent_h // patch_size[1])
            base_w = max(1, latent_w // patch_size[2])
            if base_h * base_w == hw:
                h, w = base_h, base_w
            else:
                scale = max(1, int(round((base_h * base_w) / hw)))
                shrink = int(round(scale ** 0.5))
                h = max(1, base_h // shrink)
                w = max(1, base_w // shrink)
                if h * w != hw:
                    raise ValueError(f"Cannot resolve grid size for seq_len={seq_len}, f={f}")
        return (f, h, w)

    def _time_embed(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.dim() > 1:
            timestep = timestep.flatten()[:1]
        return self.pipe.dit.time_embedding(
            sinusoidal_embedding_1d(self.pipe.dit.freq_dim, timestep)
        )

    def predict_motion(self, features: torch.Tensor, timestep: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        if self.motion_head is None:
            return None
        t_embed = self._time_embed(timestep)
        motion_pred = self.motion_head(features, t_embed)
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size
        motion_pred = rearrange(
            motion_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
            c=self.motion_channels,
        )
        return motion_pred

    def predict_depth(self, features: torch.Tensor, timestep: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        if self.depth_head is None:
            return None
        t_embed = self._time_embed(timestep)
        depth_pred = self.depth_head(features, t_embed)
        f, h, w = grid_size
        patch_size = self.pipe.dit.patch_size
        depth_pred = rearrange(
            depth_pred, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=patch_size[0], y=patch_size[1], z=patch_size[2],
            c=1,
        )
        return depth_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_image: Image.Image,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 480,
        num_frames: int = 49,
        seed: int = 42,
        num_inference_steps: int = 50,
        cfg_scale: float = 7.0,
        tiled: bool = False,
        return_motion: bool = False,
        return_depth: bool = False,
    ):
        self._captured_features = None
        self._captured_timestep = None

        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            tiled=tiled,
        )

        result = {"video": video}
        if self._captured_features is None:
            return result

        timestep = self._captured_timestep
        if timestep is None:
            timestep = torch.tensor([0.0], device=self.pipe.device, dtype=self.pipe.torch_dtype)
        grid_size = self._grid_size(num_frames, height, width, self._captured_features.shape[1])

        if return_motion:
            motion = self.predict_motion(self._captured_features, timestep, grid_size)
            if motion is not None:
                motion = motion.float().cpu().numpy()
            result["motion_vectors"] = motion
        if return_depth:
            depth = self.predict_depth(self._captured_features, timestep, grid_size)
            if depth is not None:
                depth = depth.float().cpu().numpy()
            result["depth_maps"] = depth
        return result


def load_motion_head(
    checkpoint_path: str,
    dim: int,
    motion_channels: int,
    patch_size: Tuple[int, int, int],
    device: str,
    dtype: torch.dtype,
) -> Optional[MotionVectorHead]:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    motion_head = MotionVectorHead(
        dim=dim,
        motion_channels=motion_channels,
        patch_size=patch_size,
        eps=1e-6,
        output_scale=1.0,
    )
    state_dict = load_state_dict(checkpoint_path)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("motion_head."):
            cleaned[key[len("motion_head."):]] = value
        elif key.startswith("head.") and "motion" not in key:
            continue
        else:
            cleaned[key] = value
    if any(k.startswith("motion_head.") for k in state_dict):
        state_dict = cleaned
    motion_head.load_state_dict(state_dict, strict=False)
    return motion_head.to(device=device, dtype=dtype)


def load_depth_head(
    checkpoint_path: str,
    dim: int,
    patch_size: Tuple[int, int, int],
    device: str,
    dtype: torch.dtype,
) -> Optional[DepthHead]:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None
    depth_head = DepthHead(
        dim=dim,
        depth_channels=1,
        patch_size=patch_size,
        eps=1e-6,
        output_scale=1.0,
    )
    state_dict = load_state_dict(checkpoint_path)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("depth_head."):
            cleaned[key[len("depth_head."):]] = value
        elif key.startswith("head.") and "depth" not in key:
            continue
        else:
            cleaned[key] = value
    if any(k.startswith("depth_head.") for k in state_dict):
        state_dict = cleaned
    depth_head.load_state_dict(state_dict, strict=False)
    return depth_head.to(device=device, dtype=dtype)


def resolve_wan22_paths(model_base_path: str) -> Tuple[str, str, str, str]:
    wan22_root = f"{model_base_path}/Wan2.2-TI2V-5B"
    wan22_dit_dir = wan22_root
    nested_dir = f"{wan22_root}/Wan-AI/Wan2___2-TI2V-5B"
    if Path(nested_dir).is_dir():
        wan22_dit_dir = nested_dir
    t5_path = f"{wan22_root}/models_t5_umt5-xxl-enc-bf16.pth"
    if not Path(t5_path).exists() and Path(nested_dir).is_dir():
        t5_path = f"{nested_dir}/models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = f"{wan22_root}/Wan2.2_VAE.pth"
    if not Path(vae_path).exists() and Path(nested_dir).is_dir():
        vae_path = f"{nested_dir}/Wan2.2_VAE.pth"
    tokenizer_dir = f"{wan22_root}/google/umt5-xxl"
    if not Path(tokenizer_dir).exists() and Path(nested_dir).is_dir():
        tokenizer_dir = f"{nested_dir}/google/umt5-xxl"
    return wan22_dit_dir, t5_path, vae_path, tokenizer_dir


def resolve_wan22_dit_path(wan22_dit_dir: str):
    shard_pattern = os.path.join(wan22_dit_dir, "diffusion_pytorch_model*.safetensors")
    shards = sorted(glob.glob(shard_pattern))
    if shards:
        return shards
    return wan22_dit_dir


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = np.mod(i, 6)

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

    mask = i == 0
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]
    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]
    mask = i == 2
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]
    mask = i == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]
    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]
    mask = i == 5
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def motion_vectors_to_video(motion_vectors: np.ndarray):
    if motion_vectors is None:
        return None
    motion = motion_vectors[0]
    dx = motion[0]
    dy = motion[1]
    mag = np.sqrt(dx * dx + dy * dy)
    max_mag = mag.max() if mag.size else 1.0
    if max_mag <= 0:
        max_mag = 1.0
    mag = np.clip(mag / max_mag, 0.0, 1.0)
    ang = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)

    frames = []
    for t in range(motion.shape[1]):
        h = ang[t]
        s = np.ones_like(h)
        v = mag[t]
        rgb = _hsv_to_rgb(h, s, v)
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(rgb, mode="RGB"))
    return frames


def depth_maps_to_video(depth_maps: np.ndarray):
    if depth_maps is None:
        return None
    depth = depth_maps[0, 0]
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    frames = []
    for t in range(depth.shape[0]):
        frame = (depth[t] * 255.0).clip(0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame, mode="L").convert("RGB"))
    return frames


def main():
    parser = argparse.ArgumentParser(description="WAN2.2 TI2V Stage 2 LoRA inference")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_base_path", type=str, default="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI")
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--motion_head_checkpoint", type=str, default="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage2/final/motion_head.pth")
    parser.add_argument("--depth_head_checkpoint", type=str, default="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage2/final/depth_head.pth")
    parser.add_argument("--output_motion", type=str, default=None)
    parser.add_argument("--output_depth", type=str, default=None)
    parser.add_argument("--output_motion_video", type=str, default=None)
    parser.add_argument("--output_depth_video", type=str, default=None)
    args = parser.parse_args()

    input_image = Image.open(args.input_image).convert("RGB")
    wan22_dit_dir, t5_path, vae_path, tokenizer_dir = resolve_wan22_paths(args.model_base_path)
    dit_path = resolve_wan22_dit_path(wan22_dit_dir)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=dit_path, offload_device="cpu"),
            ModelConfig(path=t5_path, offload_device="cpu"),
            ModelConfig(path=vae_path, offload_device="cpu"),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_dir),
    )

    if args.lora_checkpoint and Path(args.lora_checkpoint).exists():
        lora_state_dict = load_state_dict(args.lora_checkpoint)
        lora_loader = GeneralLoRALoader(device=args.device, torch_dtype=torch.bfloat16)
        lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)

    pipe.enable_vram_management()

    motion_head = load_motion_head(
        checkpoint_path=args.motion_head_checkpoint,
        dim=pipe.dit.dim,
        motion_channels=4,
        patch_size=pipe.dit.patch_size,
        device=args.device,
        dtype=torch.bfloat16,
    )
    depth_head = load_depth_head(
        checkpoint_path=args.depth_head_checkpoint,
        dim=pipe.dit.dim,
        patch_size=pipe.dit.patch_size,
        device=args.device,
        dtype=torch.bfloat16,
    )

    motion_depth_pipe = MotionDepthWanPipeline(
        pipe=pipe,
        motion_head=motion_head,
        depth_head=depth_head,
        motion_channels=4,
    )

    result = motion_depth_pipe(
        prompt=args.prompt,
        input_image=input_image,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        return_motion=args.output_motion is not None or args.output_motion_video is not None,
        return_depth=args.output_depth is not None or args.output_depth_video is not None,
    )

    save_video(result["video"], args.output, fps=args.fps, quality=5)

    if args.output_motion and result.get("motion_vectors") is not None:
        np.save(args.output_motion, result["motion_vectors"])
    if args.output_depth and result.get("depth_maps") is not None:
        np.save(args.output_depth, result["depth_maps"])
    if args.output_motion_video and result.get("motion_vectors") is not None:
        motion_frames = motion_vectors_to_video(result["motion_vectors"])
        if motion_frames is not None:
            save_video(motion_frames, args.output_motion_video, fps=args.fps, quality=5)
    if args.output_depth_video and result.get("depth_maps") is not None:
        depth_frames = depth_maps_to_video(result["depth_maps"])
        if depth_frames is not None:
            save_video(depth_frames, args.output_depth_video, fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
