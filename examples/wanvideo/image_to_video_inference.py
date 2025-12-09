"""Command line helper to run WanVideo image-to-video inference.

This script wraps the example pipelines shipped with DiffSynth-Studio so you can
swap Wan I2V checkpoints, provide a custom prompt, and generate an mp4 from a
single reference image. It mirrors the structure of the example scripts under
"examples/wanvideo/model_inference" but exposes a more ergonomic CLI. Batch mode
saves results beneath a hidden directory named after the model (e.g. `.wan2.2`).
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


@dataclass(frozen=True)
class Variant:
    """Metadata describing how to load each Wan I2V checkpoint."""

    model_id: str
    resources: tuple[str, ...]
    default_height: int | None = None
    default_width: int | None = None
    description: str | None = None

    def to_model_configs(
        self,
        offload_device: str | None,
        local_model_root: Path | None = None,
    ) -> list[ModelConfig]:
        local_model_path = str(local_model_root) if local_model_root else None
        skip_download = local_model_root is not None
        return [
            ModelConfig(
                model_id=self.model_id,
                origin_file_pattern=pattern,
                offload_device=offload_device,
                local_model_path=local_model_path,
                skip_download=skip_download,
            )
            for pattern in self.resources
        ]


# File patterns follow the upstream examples inside model_inference/.
MODEL_VARIANTS: dict[str, Variant] = {
    "wan2.2-i2v-a14b": Variant(
        model_id="Wan-AI/Wan2.2-I2V-A14B",
        resources=(
            "high_noise_model/diffusion_pytorch_model*.safetensors",
            "low_noise_model/diffusion_pytorch_model*.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
        ),
        default_height=480,
        default_width=832,
        description="Wan 2.2 I2V A14B, default 480x832",
    ),
    "wan2.1-i2v-14b-480p": Variant(
        model_id="Wan-AI/Wan2.1-I2V-14B-480P",
        resources=(
            "diffusion_pytorch_model*.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ),
        default_height=480,
        default_width=832,
        description="Wan 2.1 I2V 14B checkpoint tuned for 480p",
    ),
    "wan2.1-i2v-14b-720p": Variant(
        model_id="Wan-AI/Wan2.1-I2V-14B-720P",
        resources=(
            "diffusion_pytorch_model*.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ),
        default_height=720,
        default_width=1280,
        description="Wan 2.1 I2V 14B checkpoint tuned for 720p",
    ),
}


SWITCH_NAME_RE = re.compile(r"^(?P<idx>\d+)_switch-frames(?:_[^_]+)?_(?P<rest>.+)$")
MODEL_DIR_SANITIZE_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def image_to_generated_name(image_path: Path) -> str:
    """Map a switch-frame still image name to its generated video filename."""

    stem = image_path.stem
    match = SWITCH_NAME_RE.match(stem)
    if match:
        return f"{match['idx']}_{match['rest']}.mp4"
    return f"{stem}.mp4"


def sanitize_model_tag(tag: str) -> str:
    cleaned = MODEL_DIR_SANITIZE_RE.sub("-", tag.strip()) or "model"
    return cleaned


def model_output_dir(base_dir: Path, model_tag: str) -> Path:
    normalized = sanitize_model_tag(model_tag)
    if not normalized.startswith('.'):
        normalized = f".{normalized}"
    return base_dir / normalized


def load_descriptions(csv_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("generated_video_name")
            description = row.get("description", "")
            if not key:
                continue
            lookup[key] = description.strip()
    return lookup


def gather_images(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in suffixes and path.is_file()
    )


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return ivalue


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wan image-to-video inference with a single command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a single image or a directory containing images.",
    )
    parser.add_argument(
        "--variant",
        default="wan2.2-i2v-a14b",
        choices=sorted(MODEL_VARIANTS),
        help="Which pretrained Wan I2V model bundle to use.",
    )
    parser.add_argument("--prompt", default="", help="Positive text prompt driving the motion.")
    parser.add_argument("--negative-prompt", default="", help="Negative text prompt.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--height",
        type=positive_int,
        default=None,
        help="Target video height. Defaults to the variant's recommended size.",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=None,
        help="Target video width. Defaults to the variant's recommended size.",
    )
    parser.add_argument(
        "--resize-image",
        action="store_true",
        help="Resize the input image to match the requested height/width.",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=15,
        help="Frames per second for the saved video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output video path for single-image mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory for outputs; a .model_tag folder is created inside.",
    )
    parser.add_argument(
        "--description-csv",
        type=Path,
        default=None,
        help="Optional CSV file containing columns generated_video_name and description used to build prompts.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation for inputs whose output file already exists.",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Custom tag for the output directory (defaults to the chosen variant).",
    )
    parser.add_argument(
        "--local-model-root",
        type=Path,
        default=None,
        help="Directory containing locally downloaded Wan checkpoints (mirrors ModelScope layout).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for the pipeline (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype for the loaded weights.",
    )
    parser.add_argument(
        "--offload-device",
        default="cpu",
        help="Device used for modules that are offloaded from the main device.",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiling to reduce memory usage during generation.",
    )
    parser.add_argument(
        "--switch-dit-boundary",
        type=float,
        default=None,
        help="Override switch_DiT_boundary (Wan 2.2 only).",
    )
    parser.add_argument(
        "--disable-vram-management",
        dest="enable_vram_management",
        action="store_false",
        help="Opt out of pipeline VRAM management helpers.",
    )
    parser.set_defaults(enable_vram_management=True)
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    return Image.open(path).convert("RGB")


def get_dtype(name: str) -> torch.dtype:
    lookup = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return lookup[name]


def maybe_resize(image: Image.Image, width: int | None, height: int | None) -> Image.Image:
    if width is None or height is None:
        return image
    return image.resize((width, height), Image.BICUBIC)


def init_pipeline(args: argparse.Namespace, variant: Variant) -> WanVideoPipeline:
    local_root = args.local_model_root
    if local_root is not None and not local_root.exists():
        raise FileNotFoundError(f"Local model root not found: {local_root}")

    tokenizer_config = None
    if local_root is not None:
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/*",
            local_model_path=str(local_root),
            skip_download=True,
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=get_dtype(args.dtype),
        device=args.device,
        model_configs=variant.to_model_configs(args.offload_device, local_root),
        tokenizer_config=tokenizer_config if tokenizer_config is not None else ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
    )
    if args.enable_vram_management:
        pipe.enable_vram_management()
    return pipe


def resolve_prompt(
    image_path: Path,
    args: argparse.Namespace,
    descriptions: dict[str, str],
) -> str:
    prompt = args.prompt
    if descriptions:
        key = image_to_generated_name(image_path)
        description = descriptions.get(key)
        if description:
            prompt = description
        else:
            print(f"No description found for {image_path.name}; using CLI prompt.")
    return prompt


def run_single(
    args: argparse.Namespace,
    variant: Variant,
    image_path: Path,
    width: int | None,
    height: int | None,
    descriptions: dict[str, str],
    model_dir: Path,
) -> None:
    prompt = resolve_prompt(image_path, args, descriptions)

    output_path = args.output
    if output_path is None:
        model_dir.mkdir(parents=True, exist_ok=True)
        output_name = image_to_generated_name(image_path)
        output_path = model_dir / output_name
    if args.skip_existing and output_path.exists():
        print(f"Skipping {output_path}; already exists.")
        return

    input_image = load_image(image_path)
    if args.resize_image and width and height:
        input_image = maybe_resize(input_image, width, height)

    pipe = init_pipeline(args, variant)

    generation_kwargs = dict(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        input_image=input_image,
        tiled=args.tiled,
    )
    if height is not None:
        generation_kwargs["height"] = height
    if width is not None:
        generation_kwargs["width"] = width
    if args.switch_dit_boundary is not None:
        generation_kwargs["switch_DiT_boundary"] = args.switch_dit_boundary

    video = pipe(**generation_kwargs)

    ensure_parent(output_path)
    save_video(video, str(output_path), fps=args.fps, quality=5)


def run_batch(
    args: argparse.Namespace,
    variant: Variant,
    directory: Path,
    width: int | None,
    height: int | None,
    descriptions: dict[str, str],
    model_dir: Path,
) -> None:
    images = gather_images(directory)
    if not images:
        raise ValueError(f"No images found in {directory}")

    model_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path]] = []
    for image_path in images:
        output_name = image_to_generated_name(image_path)
        output_path = model_dir / output_name
        if args.skip_existing and output_path.exists():
            print(f"Skipping {output_path.name}; already exists.")
            continue
        tasks.append((image_path, output_path))

    if not tasks:
        print("All outputs already exist; nothing to do.")
        return

    pipe = init_pipeline(args, variant)

    for image_path, output_path in tasks:
        prompt = resolve_prompt(image_path, args, descriptions)

        input_image = load_image(image_path)
        if args.resize_image and width and height:
            input_image = maybe_resize(input_image, width, height)

        generation_kwargs = dict(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            input_image=input_image,
            tiled=args.tiled,
        )
        if height is not None:
            generation_kwargs["height"] = height
        if width is not None:
            generation_kwargs["width"] = width
        if args.switch_dit_boundary is not None:
            generation_kwargs["switch_DiT_boundary"] = args.switch_dit_boundary

        video = pipe(**generation_kwargs)

        ensure_parent(output_path)
        save_video(video, str(output_path), fps=args.fps, quality=5)
        print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()

    variant = MODEL_VARIANTS[args.variant]

    width = args.width or variant.default_width
    height = args.height or variant.default_height

    descriptions = load_descriptions(args.description_csv) if args.description_csv else {}

    base_output_dir = args.output_dir or Path(".")
    model_tag = args.model_tag or args.variant
    output_dir = model_output_dir(base_output_dir, model_tag)

    input_path = args.input_path
    if input_path.is_dir():
        run_batch(args, variant, input_path, width, height, descriptions, output_dir)
    elif input_path.is_file():
        run_single(args, variant, input_path, width, height, descriptions, output_dir)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
