#!/usr/bin/env python3
"""
Physics-IQ Benchmark Inference Pipeline for WAN2.2 Model
Processes images and prompts from the physics-IQ-benchmark dataset
"""

import os
import csv
import torch
import time
from pathlib import Path
from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from tqdm import tqdm
import argparse
from PIL import Image

# Model paths
BASE_MODEL_PATH = '/nyx-storage1/hanliu/world_model_ckpt'
PHYSICS_IQ_PATH = '/nyx-storage1/hanliu/physics-IQ-benchmark'
SWITCH_FRAMES_PATH = os.path.join(PHYSICS_IQ_PATH, 'physics-IQ-benchmark/switch-frames')
DESCRIPTIONS_PATH = os.path.join(PHYSICS_IQ_PATH, 'descriptions/descriptions.csv')

def load_prompts_and_images(descriptions_path, switch_frames_path, max_samples=None):
    """Load prompts and corresponding image paths from CSV."""
    data = []

    with open(descriptions_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break

            # Extract video ID and perspective from the scenario column
            scenario = row['scenario']
            # Get the ID number (first 4 digits)
            video_id = scenario[:4]

            # Extract perspective and scenario name from the generated_video_name
            generated_name = row.get('generated_video_name', '')
            # Format: 0001_perspective-left_trimmed-ball-and-block-fall.mp4
            parts = generated_name.replace('.mp4', '').split('_', 2)
            perspective = parts[1] if len(parts) > 1 else 'unknown'
            scenario_name = parts[2] if len(parts) > 2 else 'unknown'

            # Find corresponding switch-frame image
            # The pattern is: XXXX_switch-frames_anyFPS_perspective-{left|center|right}_trimmed-*.jpg
            image_pattern = f"{video_id}_switch-frames"

            # Find matching image files
            matching_images = []
            for img_file in os.listdir(switch_frames_path):
                if img_file.startswith(image_pattern) and perspective in img_file:
                    matching_images.append(os.path.join(switch_frames_path, img_file))

            # If no perspective-specific match, try any matching image
            if not matching_images:
                for img_file in os.listdir(switch_frames_path):
                    if img_file.startswith(image_pattern):
                        matching_images.append(os.path.join(switch_frames_path, img_file))
                        break

            if matching_images:
                # Use the first matching image
                data.append({
                    'video_id': video_id,
                    'perspective': perspective,
                    'scenario_name': scenario_name,
                    'scenario': scenario,
                    'prompt': row['description'],
                    'category': row['category'],
                    'image_path': matching_images[0],
                    'image_filename': os.path.basename(matching_images[0]),
                    'output_name': generated_name
                })
            # Silently skip scenarios without switch-frame images
            # (switch-frames only available for scenarios 0001-0198)

    return data

def load_wan22_pipeline(device="cuda"):
    """Load WAN2.2-TI2V-5B model pipeline (using local checkpoints)."""
    print("Loading Wan2.2-TI2V-5B model from local checkpoints...")
    print("Using local checkpoint files from:", BASE_MODEL_PATH)

    # Use Wan2.2-TI2V-5B model (Text+Image-to-Video model that supports input_image)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                       origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                       offload_device="cpu",
                       local_model_path=BASE_MODEL_PATH),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                       origin_file_pattern="diffusion_pytorch_model*.safetensors",
                       offload_device="cpu",
                       local_model_path=BASE_MODEL_PATH),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                       origin_file_pattern="Wan2.2_VAE.pth",
                       offload_device="cpu",
                       local_model_path=BASE_MODEL_PATH),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                                    origin_file_pattern="google/*",
                                    local_model_path=BASE_MODEL_PATH),
    )

    pipe.enable_vram_management()
    return pipe

def process_single_sample(pipe, sample, output_dir, video_config, model_name="wan22"):
    """Process a single sample with image and prompt."""

    print(f"\nProcessing {sample['video_id']}: {sample['scenario'][:50]}...")
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"Image: {sample['image_path']}")

    # Load the input image
    input_image = Image.open(sample['image_path']).convert('RGB')

    # Resize image to match model requirements if needed
    if video_config['resize_input']:
        input_image = input_image.resize((video_config['width'], video_config['height']), Image.LANCZOS)

    start_time = time.time()

    try:
        # Generate video with image-to-video
        video = pipe(
            prompt=sample['prompt'],
            negative_prompt=video_config['negative_prompt'],
            input_image=input_image,  # Add input image for I2V
            height=video_config['height'],
            width=video_config['width'],
            num_frames=video_config['num_frames'],
            cfg_scale=video_config['cfg_scale'],
            num_inference_steps=video_config['num_inference_steps'],
            seed=video_config['seed'],
            tiled=video_config['tiled'],
        )

        generation_time = time.time() - start_time

        # Save video following Physics-IQ format: .model_name/{ID}_{perspective}_{scenario_name}.mp4
        model_dir = output_dir / f".{model_name}"
        model_dir.mkdir(exist_ok=True)

        # Format: {ID}_{perspective}_{scenario_name}.mp4
        output_filename = f"{sample['video_id']}_{sample['perspective']}_{sample['scenario_name']}.mp4"
        output_path = model_dir / output_filename

        save_video(video, str(output_path), fps=video_config['fps'], quality=video_config['quality'])

        print(f"✓ Generated in {generation_time:.2f}s -> {output_path}")

        return {
            'success': True,
            'video_id': sample['video_id'],
            'output_path': str(output_path),
            'generation_time': generation_time
        }

    except Exception as e:
        print(f"✗ Failed to generate: {str(e)}")
        return {
            'success': False,
            'video_id': sample['video_id'],
            'error': str(e)
        }

def generate_report(results, output_dir, total_time, shard_index=None, num_shards=1):
    """Generate a summary report of the inference results."""
    if shard_index is not None:
        report_path = output_dir / f"physics_iq_inference_report_shard{shard_index}.md"
    else:
        report_path = output_dir / "physics_iq_inference_report.md"

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    with open(report_path, "w") as f:
        if shard_index is not None:
            f.write(f"# Physics-IQ Wan2.2-TI2V-5B Inference Report - Shard {shard_index}/{num_shards}\n\n")
        else:
            f.write("# Physics-IQ Wan2.2-TI2V-5B Inference Report\n\n")
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

def main():
    parser = argparse.ArgumentParser(description="Physics-IQ Wan2.2-TI2V-5B Inference Pipeline")
    parser.add_argument("--output_dir", type=str, default="physics_iq_results",
                       help="Output directory for generated videos")
    parser.add_argument("--model_name", type=str, default="wan22_ti2v_5b",
                       help="Model name for output directory (creates .model_name folder)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (None for all)")
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--width", type=int, default=832,
                       help="Video width")
    parser.add_argument("--num_frames", type=int, default=81,
                       help="Number of frames to generate (81 frames = ~5 seconds at 16fps)")
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of denoising steps")
    parser.add_argument("--fps", type=int, default=16,
                       help="Output video FPS (benchmark evaluates first 5 seconds)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--resize_input", action="store_true",
                       help="Resize input images to match output dimensions")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--shard_index", type=int, default=None,
                       help="Shard index (0 to num_shards-1) for multi-GPU processing")
    parser.add_argument("--num_shards", type=int, default=1,
                       help="Total number of shards for multi-GPU processing")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("Physics-IQ Wan2.2-TI2V-5B Inference Pipeline")
    print("="*60)

    # Video configuration
    video_config = {
        'height': args.height,
        'width': args.width,
        'num_frames': args.num_frames,
        'cfg_scale': args.cfg_scale,
        'num_inference_steps': args.num_inference_steps,
        'fps': args.fps,
        'seed': args.seed,
        'quality': 5,
        'tiled': True,
        'resize_input': args.resize_input,
        'negative_prompt': "blurry, low quality, distorted, static image, cartoon, animation"
    }

    print("\nConfiguration:")
    for key, value in video_config.items():
        print(f"  {key}: {value}")

    # Load data
    print(f"\nLoading data from:")
    print(f"  Images: {SWITCH_FRAMES_PATH}")
    print(f"  Descriptions: {DESCRIPTIONS_PATH}")

    data = load_prompts_and_images(DESCRIPTIONS_PATH, SWITCH_FRAMES_PATH, args.max_samples)
    print(f"\nFound {len(data)} valid samples with switch-frame images to process")
    print(f"(Note: Only scenarios 0001-0198 have switch-frame images available)")

    if not data:
        print("No data found to process!")
        return

    # Shard data for multi-GPU processing
    if args.shard_index is not None:
        total_samples = len(data)
        samples_per_shard = total_samples // args.num_shards
        extra_samples = total_samples % args.num_shards

        # Calculate start and end indices for this shard
        if args.shard_index < extra_samples:
            # First 'extra_samples' shards get one additional sample
            start_idx = args.shard_index * (samples_per_shard + 1)
            end_idx = start_idx + samples_per_shard + 1
        else:
            start_idx = args.shard_index * samples_per_shard + extra_samples
            end_idx = start_idx + samples_per_shard

        data = data[start_idx:end_idx]
        print(f"\n🔀 Shard {args.shard_index}/{args.num_shards}: Processing samples {start_idx} to {end_idx-1} ({len(data)} samples)")

    if not data:
        print("No data assigned to this shard!")
        return

    # Load model
    pipe = load_wan22_pipeline(device=args.device)

    # Process samples
    print(f"\nStarting inference on {len(data)} samples...")
    print("="*60)

    results = []
    total_start_time = time.time()

    for sample in tqdm(data, desc="Processing samples"):
        result = process_single_sample(pipe, sample, output_dir, video_config, args.model_name)
        results.append(result)

    total_time = time.time() - total_start_time

    # Generate report
    generate_report(results, output_dir, total_time, args.shard_index, args.num_shards)

    print("\n" + "="*60)
    print("✅ Inference Complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Model output directory: {output_dir / f'.{args.model_name}'}")
    print(f"Total time: {total_time:.2f}s")

    # Summary statistics
    successful = sum(1 for r in results if r['success'])
    print(f"Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")

    print("\n📌 Physics-IQ Benchmark Notes:")
    print(f"- Videos saved in: {output_dir}/.{args.model_name}/")
    print("- Format: {ID}_{perspective}_{scenario_name}.mp4")
    print("- First 5 seconds of each video will be evaluated")
    print("- To run evaluation, use the benchmark's evaluation scripts")

if __name__ == "__main__":
    main()