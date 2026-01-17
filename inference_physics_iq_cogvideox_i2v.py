#!/usr/bin/env python3
"""
Physics-IQ Benchmark Inference Pipeline for CogVideoX-5b-I2V (diffusers).
Processes images and prompts from the physics-IQ-benchmark dataset.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from tqdm import tqdm


DEFAULT_PHYSICS_IQ_ROOT = os.environ.get("PHYSICS_IQ_ROOT", "/nyx-storage1/hanliu/physics-IQ-benchmark")
DEFAULT_SWITCH_FRAMES = os.path.join(DEFAULT_PHYSICS_IQ_ROOT, "physics-IQ-benchmark/switch-frames")
DEFAULT_DESCRIPTIONS = os.path.join(DEFAULT_PHYSICS_IQ_ROOT, "descriptions/descriptions.csv")
DEFAULT_MODEL_ID = "THUDM/CogVideoX-5b-I2V"


def load_prompts_and_images(descriptions_path, switch_frames_path, max_samples=None):
    """Load prompts and corresponding image paths from CSV."""
    data = []
    with open(descriptions_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break

            scenario = row["scenario"]
            video_id = scenario[:4]

            generated_name = row.get("generated_video_name", "")
            parts = generated_name.replace(".mp4", "").split("_", 2)
            perspective = parts[1] if len(parts) > 1 else "unknown"
            scenario_name = parts[2] if len(parts) > 2 else "unknown"

            image_pattern = f"{video_id}_switch-frames"

            matching_images = []
            for img_file in os.listdir(switch_frames_path):
                if img_file.startswith(image_pattern) and perspective in img_file:
                    matching_images.append(os.path.join(switch_frames_path, img_file))

            if not matching_images:
                for img_file in os.listdir(switch_frames_path):
                    if img_file.startswith(image_pattern):
                        matching_images.append(os.path.join(switch_frames_path, img_file))
                        break

            if matching_images:
                data.append(
                    {
                        "video_id": video_id,
                        "perspective": perspective,
                        "scenario_name": scenario_name,
                        "scenario": scenario,
                        "prompt": row["description"],
                        "category": row["category"],
                        "image_path": matching_images[0],
                        "image_filename": os.path.basename(matching_images[0]),
                        "output_name": generated_name,
                    }
                )

    return data


def build_generator(device, seed):
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def process_single_sample(pipe, sample, output_dir, video_config, model_name, device):
    print(f"\nProcessing {sample['video_id']}: {sample['scenario'][:50]}...")
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Image: {sample['image_path']}")

    input_image = load_image(sample["image_path"])
    if video_config["resize_input"]:
        input_image = input_image.resize((video_config["width"], video_config["height"]))

    generator = build_generator(device, video_config["seed"])

    start_time = time.time()
    try:
        output = pipe(
            prompt=sample["prompt"],
            image=input_image,
            num_videos_per_prompt=1,
            num_inference_steps=video_config["num_inference_steps"],
            num_frames=video_config["num_frames"],
            guidance_scale=video_config["guidance_scale"],
            generator=generator,
        )
        video = output.frames[0]
        generation_time = time.time() - start_time

        model_dir = output_dir / f".{model_name}"
        model_dir.mkdir(exist_ok=True)
        output_filename = f"{sample['video_id']}_{sample['perspective']}_{sample['scenario_name']}.mp4"
        output_path = model_dir / output_filename

        export_to_video(video, str(output_path), fps=video_config["fps"])
        print(f"✓ Generated in {generation_time:.2f}s -> {output_path}")

        return {
            "success": True,
            "video_id": sample["video_id"],
            "output_path": str(output_path),
            "generation_time": generation_time,
        }
    except Exception as e:
        print(f"✗ Failed to generate: {str(e)}")
        return {"success": False, "video_id": sample["video_id"], "error": str(e)}


def generate_report(results, output_dir, total_time, shard_index=None, num_shards=1):
    if shard_index is not None:
        report_path = output_dir / f"physics_iq_cogvideox_report_shard{shard_index}.md"
    else:
        report_path = output_dir / "physics_iq_cogvideox_report.md"

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    with open(report_path, "w") as f:
        title = "Physics-IQ CogVideoX-5b-I2V Inference Report"
        if shard_index is not None:
            title += f" - Shard {shard_index}/{num_shards}"
        f.write(f"# {title}\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n")
        if shard_index is not None:
            f.write(f"- **Shard:** {shard_index} of {num_shards}\n")
        f.write(f"- **Total samples processed:** {len(results)}\n")
        f.write(f"- **Successful:** {len(successful)}\n")
        f.write(f"- **Failed:** {len(failed)}\n")
        f.write(f"- **Total time:** {total_time:.2f}s\n")
        f.write(f"- **Average time per video:** {total_time/len(results) if results else 0:.2f}s\n\n")

        f.write("## Successful Generations\n")
        for r in successful:
            f.write(f"- {r['video_id']}: {r['output_path']} ({r['generation_time']:.2f}s)\n")

        if failed:
            f.write("\n## Failed Generations\n")
            for r in failed:
                f.write(f"- {r['video_id']}: {r['error']}\n")

    print(f"\n📊 Report saved to: {report_path}")


def shard_data(data, shard_index, num_shards):
    total_samples = len(data)
    samples_per_shard = total_samples // num_shards
    extra_samples = total_samples % num_shards

    if shard_index < extra_samples:
        start_idx = shard_index * (samples_per_shard + 1)
        end_idx = start_idx + samples_per_shard + 1
    else:
        start_idx = shard_index * samples_per_shard + extra_samples
        end_idx = start_idx + samples_per_shard

    return data[start_idx:end_idx], start_idx, end_idx


def resolve_dtype(dtype_name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def main():
    parser = argparse.ArgumentParser(description="Physics-IQ CogVideoX-5b-I2V Inference Pipeline (diffusers)")
    parser.add_argument("--output_dir", type=str, default="physics_iq_results_cogvideox_i2v")
    parser.add_argument("--model_name", type=str, default="cogvideox_i2v_5b")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--physics_iq_root", type=str, default=DEFAULT_PHYSICS_IQ_ROOT)
    parser.add_argument("--descriptions_path", type=str, default=None)
    parser.add_argument("--switch_frames_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_input", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--enable_offload", action="store_true")
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument("--vae_slicing", action="store_true")
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    descriptions_path = args.descriptions_path or os.path.join(
        args.physics_iq_root, "descriptions", "descriptions.csv"
    )
    switch_frames_path = args.switch_frames_path or os.path.join(
        args.physics_iq_root, "physics-IQ-benchmark", "switch-frames"
    )

    torch_dtype = resolve_dtype(args.dtype)

    print("=" * 60)
    print("Physics-IQ CogVideoX-5b-I2V Inference Pipeline (diffusers)")
    print("=" * 60)

    guidance_scale = args.guidance_scale if args.cfg_scale is None else args.cfg_scale

    video_config = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "guidance_scale": guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "fps": args.fps,
        "seed": args.seed,
        "resize_input": args.resize_input,
    }

    print("\nConfiguration:")
    for key, value in video_config.items():
        print(f"  {key}: {value}")

    print(f"\nModel ID: {args.model_id}")
    print(f"Device: {args.device} | Dtype: {args.dtype}")

    print("\nLoading data from:")
    print(f"  Images: {switch_frames_path}")
    print(f"  Descriptions: {descriptions_path}")

    data = load_prompts_and_images(descriptions_path, switch_frames_path, args.max_samples)
    print(f"\nFound {len(data)} valid samples with switch-frame images to process")
    print("(Note: Only scenarios 0001-0198 have switch-frame images available)")

    if not data:
        print("No data found to process!")
        return

    if args.shard_index is not None:
        data, start_idx, end_idx = shard_data(data, args.shard_index, args.num_shards)
        print(
            f"\n🔀 Shard {args.shard_index}/{args.num_shards}: "
            f"Processing samples {start_idx} to {end_idx - 1} ({len(data)} samples)"
        )

    if not data:
        print("No data assigned to this shard!")
        return

    print("\nLoading pipeline...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.enable_offload and args.device == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(args.device)

    if args.vae_tiling:
        pipe.vae.enable_tiling()
    if args.vae_slicing:
        pipe.vae.enable_slicing()

    print(f"\nStarting inference on {len(data)} samples...")
    print("=" * 60)

    results = []
    total_start_time = time.time()
    for sample in tqdm(data, desc="Processing samples"):
        result = process_single_sample(pipe, sample, output_dir, video_config, args.model_name, args.device)
        results.append(result)

    total_time = time.time() - total_start_time
    generate_report(results, output_dir, total_time, args.shard_index, args.num_shards)

    print("\n" + "=" * 60)
    print("✅ Inference Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Model output directory: {output_dir / f'.{args.model_name}'}")
    print(f"Total time: {total_time:.2f}s")

    successful = sum(1 for r in results if r["success"])
    print(f"Success rate: {successful}/{len(results)} ({100 * successful / len(results):.1f}%)")

    print("\n📌 Physics-IQ Benchmark Notes:")
    print(f"- Videos saved in: {output_dir}/.{args.model_name}/")
    print("- Format: {ID}_{perspective}_{scenario_name}.mp4")
    print("- First 5 seconds of each video will be evaluated")
    print("- To run evaluation, use the benchmark's evaluation scripts")


if __name__ == "__main__":
    main()
