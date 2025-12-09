#!/usr/bin/env python3
"""
Multi-GPU Physics-IQ Benchmark Inference Pipeline for WAN2.2 Model
Each GPU processes a separate subset of prompts (data parallelism)
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

def shard_data(data, rank, world_size):
    """Split data into shards for multi-GPU processing."""
    total_samples = len(data)
    samples_per_gpu = total_samples // world_size
    extra_samples = total_samples % world_size

    # Calculate start and end indices for this rank
    if rank < extra_samples:
        # First 'extra_samples' ranks get one additional sample
        start_idx = rank * (samples_per_gpu + 1)
        end_idx = start_idx + samples_per_gpu + 1
    else:
        start_idx = rank * samples_per_gpu + extra_samples
        end_idx = start_idx + samples_per_gpu

    shard = data[start_idx:end_idx]
    print(f"[GPU {rank}] Processing samples {start_idx} to {end_idx-1} ({len(shard)} samples)")
    return shard

def load_wan22_pipeline(device):
    """Load WAN2.2-TI2V-5B model pipeline (using local checkpoints)."""
    print(f"[{device}] Loading Wan2.2-TI2V-5B model from local checkpoints...")
    print(f"[{device}] Using local checkpoint files from: {BASE_MODEL_PATH}")

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

def process_single_sample(pipe, sample, output_dir, video_config, model_name, rank):
    """Process a single sample with image and prompt."""

    print(f"[GPU {rank}] Processing {sample['video_id']}: {sample['scenario'][:50]}...")
    print(f"[GPU {rank}] Prompt: {sample['prompt'][:100]}...")
    print(f"[GPU {rank}] Image: {sample['image_path']}")

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

        print(f"[GPU {rank}] ✓ Generated in {generation_time:.2f}s -> {output_path}")

        return {
            'success': True,
            'video_id': sample['video_id'],
            'output_path': str(output_path),
            'generation_time': generation_time,
            'rank': rank
        }

    except Exception as e:
        print(f"[GPU {rank}] ✗ Failed to generate: {str(e)}")
        return {
            'success': False,
            'video_id': sample['video_id'],
            'error': str(e),
            'rank': rank
        }

def generate_report(results, output_dir, total_time, rank):
    """Generate a summary report of the inference results for this GPU."""
    report_path = output_dir / f"physics_iq_inference_report_gpu{rank}.md"

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    with open(report_path, "w") as f:
        f.write(f"# Physics-IQ Wan2.2-TI2V-5B Inference Report - GPU {rank}\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n")
        f.write(f"- **GPU Rank:** {rank}\n")
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

    print(f"[GPU {rank}] 📊 Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Physics-IQ Wan2.2-TI2V-5B Inference Pipeline")
    parser.add_argument("--rank", type=int, required=True,
                       help="GPU rank (0, 1, 2, or 3 for 4 GPUs)")
    parser.add_argument("--world_size", type=int, default=4,
                       help="Total number of GPUs to use")
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

    args = parser.parse_args()

    # Set device for this rank
    device = f"cuda:{args.rank}"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print(f"Physics-IQ Wan2.2-TI2V-5B Inference Pipeline - GPU {args.rank}")
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

    print(f"\n[GPU {args.rank}] Configuration:")
    for key, value in video_config.items():
        print(f"  {key}: {value}")

    # Load data
    print(f"\n[GPU {args.rank}] Loading data from:")
    print(f"  Images: {SWITCH_FRAMES_PATH}")
    print(f"  Descriptions: {DESCRIPTIONS_PATH}")

    all_data = load_prompts_and_images(DESCRIPTIONS_PATH, SWITCH_FRAMES_PATH, args.max_samples)
    print(f"\n[GPU {args.rank}] Total samples available: {len(all_data)}")

    # Shard data for this GPU
    data = shard_data(all_data, args.rank, args.world_size)

    if not data:
        print(f"[GPU {args.rank}] No data assigned to this GPU!")
        return

    # Load model on the assigned GPU
    pipe = load_wan22_pipeline(device=device)

    # Process samples
    print(f"\n[GPU {args.rank}] Starting inference on {len(data)} samples...")
    print("="*60)

    results = []
    total_start_time = time.time()

    for sample in tqdm(data, desc=f"GPU {args.rank}", position=args.rank):
        result = process_single_sample(pipe, sample, output_dir, video_config, args.model_name, args.rank)
        results.append(result)

    total_time = time.time() - total_start_time

    # Generate report
    generate_report(results, output_dir, total_time, args.rank)

    print("\n" + "="*60)
    print(f"[GPU {args.rank}] ✅ Inference Complete!")
    print("="*60)
    print(f"[GPU {args.rank}] Output directory: {output_dir}")
    print(f"[GPU {args.rank}] Model output directory: {output_dir / f'.{args.model_name}'}")
    print(f"[GPU {args.rank}] Total time: {total_time:.2f}s")

    # Summary statistics
    successful = sum(1 for r in results if r['success'])
    print(f"[GPU {args.rank}] Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")

if __name__ == "__main__":
    main()
