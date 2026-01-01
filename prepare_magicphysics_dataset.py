#!/usr/bin/env python3
"""
Prepare MagicPhysics image sequence dataset for WAN video finetuning.

Expected data structure:
    base_dir/
        test_ball_and_block_collect/
            705/
                env/camera/cam_0/rgb_capture/
                    step_0000.png
                    step_0001.png
                    ...
        test_ball_collide_collect/
            338/
                env/camera/cam_0/rgb_capture/
                    step_0000.png
                    ...

Each rgb_capture directory is treated as one video sample.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_capture_dirs(
    base_dir: str,
    capture_dir: str,
    camera_name: Optional[str],
) -> List[str]:
    """Find all directories containing image sequences."""
    capture_dirs: List[str] = []
    for root, _, files in os.walk(base_dir):
        if os.path.basename(root) != capture_dir:
            continue
        if camera_name and camera_name not in Path(root).parts:
            continue
        if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in files):
            capture_dirs.append(root)
    return sorted(capture_dirs)


def count_images_in_dir(dir_path: str) -> int:
    """Count number of image files in directory."""
    count = 0
    with os.scandir(dir_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                count += 1
    return count


def list_image_files(dir_path: str) -> List[Path]:
    """List image files in a directory, sorted by name."""
    image_files = [
        entry
        for entry in Path(dir_path).iterdir()
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_files)


def normalize_scenario_name(scenario_dir: str) -> str:
    """Normalize scenario directory name into a human-readable description."""
    name = scenario_dir
    if name.startswith("test_"):
        name = name[len("test_"):]
    if name.endswith("_collect"):
        name = name[: -len("_collect")]
    return name.replace("_", " ")


def generate_prompt_from_path(
    path: str,
    base_dir: str,
    include_scene_id: bool,
) -> str:
    """
    Generate a descriptive prompt based on the MagicPhysics path structure.

    Example path:
        test_ball_and_block_collect/705/env/camera/cam_0/rgb_capture
    """
    rel_path = os.path.relpath(path, base_dir)
    parts = Path(rel_path).parts

    if not parts:
        return "A physics simulation video"

    scenario_dir = parts[0]
    scenario = normalize_scenario_name(scenario_dir)
    scene_id = parts[1] if len(parts) > 1 else None

    prompt_templates: Dict[str, str] = {
        "ball and block": "A physics simulation of a ball and block interaction",
        "ball collide": "A physics simulation showing balls colliding",
        "ball hits duck": "A physics simulation of a ball hitting a duck",
        "ball hits nothing": "A physics simulation of a ball moving with no collision",
        "ball in basket": "A physics simulation of a ball going into a basket",
        "ball ramp": "A physics simulation of a ball rolling down a ramp",
        "ball rolls on glass": "A physics simulation of a ball rolling on glass",
        "ball train": "A physics simulation of multiple balls in sequence",
    }

    base_prompt = prompt_templates.get(
        scenario,
        f"A physics simulation showing {scenario}",
    )
    if include_scene_id and scene_id:
        return f"{base_prompt} (scene {scene_id})"
    return base_prompt


def resolve_caption_device(device: str) -> int:
    """Resolve a transformers pipeline device id."""
    device = device.strip().lower()
    if device in {"cpu", "mps"}:
        return -1
    if device.startswith("cuda"):
        parts = device.split(":")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if device.isdigit():
        return int(device)
    return 0


def build_captioner(
    model_name_or_path: str,
    device: str,
    dtype: Optional[str],
):
    """Create an image-to-text captioning pipeline."""
    from transformers import pipeline
    import torch

    torch_dtype = getattr(torch, dtype) if dtype else None
    device_id = resolve_caption_device(device)
    return pipeline(
        "image-to-text",
        model=model_name_or_path,
        device=device_id,
        torch_dtype=torch_dtype,
    )


def extract_caption_text(result: object) -> Optional[str]:
    """Extract a caption string from transformers pipeline output."""
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            if "generated_text" in item:
                return str(item["generated_text"]).strip()
            if "caption" in item:
                return str(item["caption"]).strip()
    if isinstance(result, dict):
        if "generated_text" in result:
            return str(result["generated_text"]).strip()
        if "caption" in result:
            return str(result["caption"]).strip()
    return None


def caption_sequence(
    dir_path: str,
    captioner,
    caption_frames: int,
    caption_stride: int,
) -> Optional[str]:
    """Caption a sequence by sampling frames from the directory."""
    from PIL import Image

    image_files = list_image_files(dir_path)
    if not image_files:
        return None

    stride = max(caption_stride, 1)
    selected_indices = list(range(0, len(image_files), stride))[:caption_frames]
    captions: List[str] = []

    for idx in selected_indices:
        image_path = image_files[idx]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            result = captioner(image)
        caption_text = extract_caption_text(result)
        if caption_text:
            captions.append(caption_text)

    if not captions:
        return None

    if len(captions) == 1:
        return captions[0]

    unique_captions = []
    seen = set()
    for caption in captions:
        normalized = caption.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_captions.append(caption)
    return " ".join(unique_captions)


def prepare_magicphysics_dataset(
    source_dir: str,
    output_dir: str,
    capture_dir: str = "rgb_capture",
    camera_name: Optional[str] = "cam_0",
    min_frames: int = 1,
    default_prompt: Optional[str] = None,
    include_scene_id: bool = True,
    use_parent_dir: bool = True,
    caption_model: Optional[str] = None,
    caption_device: str = "cuda",
    caption_dtype: Optional[str] = "float16",
    caption_frames: int = 1,
    caption_stride: int = 20,
) -> bool:
    """Prepare MagicPhysics dataset metadata for training."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"Error: Source directory not found: {source_path}")
        return False

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Searching for image sequences in: {source_path}")
    print(f"Looking for directories named: '{capture_dir}'")
    if camera_name:
        print(f"Filtering to camera folder: '{camera_name}'")
    if caption_model:
        print(f"Captioning enabled with model: {caption_model}")

    capture_dirs = find_capture_dirs(
        base_dir=str(source_path),
        capture_dir=capture_dir,
        camera_name=camera_name,
    )

    if not capture_dirs:
        print(f"Error: No image sequence directories found in {source_path}")
        return False

    print(f"\nFound {len(capture_dirs)} image sequence directories")

    metadata: List[Dict[str, object]] = []
    captioner = None
    if caption_model:
        captioner = build_captioner(
            model_name_or_path=caption_model,
            device=caption_device,
            dtype=caption_dtype,
        )

    for dir_path in tqdm(capture_dirs, desc="Processing image sequences"):
        num_images = count_images_in_dir(dir_path)
        if num_images < min_frames:
            continue

        prompt = None
        if captioner:
            prompt = caption_sequence(
                dir_path=dir_path,
                captioner=captioner,
                caption_frames=caption_frames,
                caption_stride=caption_stride,
            )
        if not prompt:
            prompt = default_prompt or generate_prompt_from_path(
                dir_path,
                str(source_path),
                include_scene_id=include_scene_id,
            )

        video_dir = Path(dir_path).parent if use_parent_dir else Path(dir_path)
        metadata.append(
            {
                "video": os.path.abspath(str(video_dir)),
                "prompt": prompt,
                "negative_prompt": "",
                "num_images": num_images,
            }
        )

    if not metadata:
        print("Error: No valid image sequences found")
        return False

    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = output_path / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)

    print("\nDataset prepared successfully!")
    print(f"   - Metadata saved to: {metadata_csv_path}")
    print(f"   - Total samples: {len(metadata)}")
    print(
        "   - Images per sample: min={min_frames}, max={max_frames}, mean={mean_frames:.1f}".format(
            min_frames=metadata_df["num_images"].min(),
            max_frames=metadata_df["num_images"].max(),
            mean_frames=metadata_df["num_images"].mean(),
        )
    )

    print("\nFirst 3 metadata entries:")
    for i, entry in enumerate(metadata[:3]):
        print(f"   {i+1}. Video dir: {entry['video']}")
        print(f"      Prompt: {entry['prompt']}")
        print(f"      Images: {entry['num_images']}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MagicPhysics dataset for WAN video training"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the MagicPhysics root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/magic_physics_dataset",
        help="Directory to save metadata (default: data/magic_physics_dataset)",
    )
    parser.add_argument(
        "--capture_dir",
        type=str,
        default="rgb_capture",
        help="Capture folder name to treat as a sequence (default: rgb_capture)",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="cam_0",
        help="Camera folder to filter (default: cam_0). Use empty to disable.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=1,
        help="Minimum number of frames required per sample (default: 1)",
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="Default prompt to use for all samples (optional)",
    )
    parser.add_argument(
        "--no_scene_id",
        action="store_true",
        help="Disable appending scene id to the prompt",
    )
    parser.add_argument(
        "--no_parent_dir",
        action="store_true",
        help="Store capture folder itself as sequence root",
    )
    parser.add_argument(
        "--caption_model",
        type=str,
        default=None,
        help="Captioning model name or path for image-to-text (optional)",
    )
    parser.add_argument(
        "--caption_device",
        type=str,
        default="cuda",
        help="Captioning device (e.g., cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--caption_dtype",
        type=str,
        default="float16",
        help="Captioning dtype (e.g., float16, bfloat16, float32)",
    )
    parser.add_argument(
        "--caption_frames",
        type=int,
        default=1,
        help="Number of frames to caption per sequence",
    )
    parser.add_argument(
        "--caption_stride",
        type=int,
        default=20,
        help="Stride between sampled frames for captioning",
    )

    args = parser.parse_args()
    camera_name = args.camera_name if args.camera_name else None

    success = prepare_magicphysics_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        capture_dir=args.capture_dir,
        camera_name=camera_name,
        min_frames=args.min_frames,
        default_prompt=args.default_prompt,
        include_scene_id=not args.no_scene_id,
        use_parent_dir=not args.no_parent_dir,
        caption_model=args.caption_model,
        caption_device=args.caption_device,
        caption_dtype=args.caption_dtype,
        caption_frames=args.caption_frames,
        caption_stride=args.caption_stride,
    )

    if not success:
        print("\nDataset preparation failed. Please check the error messages above.")
        raise SystemExit(1)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
