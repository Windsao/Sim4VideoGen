#!/usr/bin/env python3
"""
Combine RGB, motion, and depth videos into a single 1x3 layout video.
"""

import argparse
import os
from pathlib import Path
from typing import List

from PIL import Image

from diffsynth import save_video


def load_video_frames(path: str) -> List[Image.Image]:
    import imageio.v2 as iio

    reader = iio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(Image.fromarray(frame))
    return frames


def resize_to_match(frames: List[Image.Image], size: tuple[int, int]) -> List[Image.Image]:
    return [frame.resize(size) for frame in frames]


def resample_frames(frames: List[Image.Image], target_len: int) -> List[Image.Image]:
    if not frames or target_len <= 0:
        return []
    if len(frames) == target_len:
        return frames
    if len(frames) == 1:
        return [frames[0] for _ in range(target_len)]
    resampled = []
    for i in range(target_len):
        idx = int(round(i * (len(frames) - 1) / (target_len - 1)))
        resampled.append(frames[idx])
    return resampled


def combine_triplets(
    rgb_frames: List[Image.Image],
    motion_frames: List[Image.Image],
    depth_frames: List[Image.Image],
) -> List[Image.Image]:
    if not rgb_frames or not motion_frames or not depth_frames:
        return []
    num_frames = len(rgb_frames)
    w, h = rgb_frames[0].size
    motion_frames = resize_to_match(resample_frames(motion_frames, num_frames), (w, h))
    depth_frames = resize_to_match(resample_frames(depth_frames, num_frames), (w, h))

    combined = []
    for i in range(num_frames):
        canvas = Image.new("RGB", (w * 3, h))
        canvas.paste(rgb_frames[i], (0, 0))
        canvas.paste(motion_frames[i], (w, 0))
        canvas.paste(depth_frames[i], (w * 2, 0))
        combined.append(canvas)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Combine RGB/motion/depth videos into 1x3 layout.")
    parser.add_argument("--rgb_video", required=True, help="Path to RGB video")
    parser.add_argument("--motion_video", required=True, help="Path to motion visualization video")
    parser.add_argument("--depth_video", required=True, help="Path to depth visualization video")
    parser.add_argument("--output_dir", default=None, help="Output directory (defaults to ./combined under rgb folder)")
    parser.add_argument("--output_name", default=None, help="Output filename (defaults to rgb basename + _combined.mp4)")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    rgb_frames = load_video_frames(args.rgb_video)
    motion_frames = load_video_frames(args.motion_video)
    depth_frames = load_video_frames(args.depth_video)

    combined_frames = combine_triplets(rgb_frames, motion_frames, depth_frames)
    if not combined_frames:
        raise RuntimeError("No frames to combine.")

    rgb_path = Path(args.rgb_video)
    output_dir = args.output_dir or str(rgb_path.parent / "combined")
    os.makedirs(output_dir, exist_ok=True)

    if args.output_name is None:
        output_name = f"{rgb_path.stem}_combined.mp4"
    else:
        output_name = args.output_name

    output_path = str(Path(output_dir) / output_name)
    save_video(combined_frames, output_path, fps=args.fps, quality=5)
    print(f"Saved combined video: {output_path}")


if __name__ == "__main__":
    main()
