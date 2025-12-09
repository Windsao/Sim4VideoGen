#!/usr/bin/env python3
"""
Multi-GPU PhyGenBench Inference Pipeline for WAN2.2 Model
Each GPU processes a separate subset of prompts (data parallelism)
"""

import os
import json
import torch
import time
from pathlib import Path
from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from tqdm import tqdm
import argparse
import re

# Model paths
BASE_MODEL_PATH = '/nyx-storage1/hanliu/world_model_ckpt'

def slugify(text, max_length=50):
    """Convert text to a filename-safe slug."""
    # Convert to lowercase and replace spaces with hyphens
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug[:max_length].strip('-')

def load_prompts(json_path, max_samples=None):
    """Load prompts from PhyGenBench JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Add index to each prompt
    prompts = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break

        prompts.append({
            'index': i,
            'caption': item['caption'],
            'physical_laws': item['physical_laws'],
            'sub_category': item['sub_category'],
            'main_category': item['main_category'],
            'slug': slugify(item['caption'])
        })

    return prompts

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

def load_wan22_pipeline(device, use_t2v=True):
    """Load WAN2.2 model pipeline (T2V or TI2V)."""
    model_id = "Wan-AI/Wan2.2-T2V-5B" if use_t2v else "Wan-AI/Wan2.2-TI2V-5B"
    print(f"[{device}] Loading {model_id} model from local checkpoints...")
    print(f"[{device}] Using local checkpoint files from: {BASE_MODEL_PATH}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id,
                       origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                       offload_device="cpu",
                       local_model_path=BASE_MODEL_PATH),
            ModelConfig(model_id=model_id,
                       origin_file_pattern="diffusion_pytorch_model*.safetensors",
                       offload_device="cpu",
                       local_model_path=BASE_MODEL_PATH),
            ModelConfig(model_id=model_id,
                       origin_file_pattern="Wan2.2_VAE.pth",
                       offload_device="cpu",
                       local_model_path=BASE_MODEL_PATH),
        ],
        tokenizer_config=ModelConfig(model_id=model_id,
                                    origin_file_pattern="google/*",
                                    local_model_path=BASE_MODEL_PATH),
    )

    pipe.enable_vram_management()
    return pipe

def process_single_sample(pipe, sample, output_dir, video_config, model_name, rank):
    """Process a single sample with prompt (text-to-video)."""

    print(f"[GPU {rank}] Processing #{sample['index']:04d}: {sample['caption'][:80]}...")
    print(f"[GPU {rank}] Category: {sample['main_category']} > {sample['sub_category']}")

    start_time = time.time()

    try:
        # Generate video from text only (T2V)
        video = pipe(
            prompt=sample['caption'],
            negative_prompt=video_config['negative_prompt'],
            height=video_config['height'],
            width=video_config['width'],
            num_frames=video_config['num_frames'],
            cfg_scale=video_config['cfg_scale'],
            num_inference_steps=video_config['num_inference_steps'],
            seed=video_config['seed'],
            tiled=video_config['tiled'],
        )

        generation_time = time.time() - start_time

        # Save video: .model_name/{index:04d}_{slug}.mp4
        model_dir = output_dir / f".{model_name}"
        model_dir.mkdir(exist_ok=True)

        # Format: {index:04d}_{slug}.mp4
        output_filename = f"{sample['index']:04d}_{sample['slug']}.mp4"
        output_path = model_dir / output_filename

        save_video(video, str(output_path), fps=video_config['fps'], quality=video_config['quality'])

        print(f"[GPU {rank}] ✓ Generated in {generation_time:.2f}s -> {output_path}")

        return {
            'success': True,
            'index': sample['index'],
            'caption': sample['caption'],
            'category': f"{sample['main_category']}/{sample['sub_category']}",
            'output_path': str(output_path),
            'generation_time': generation_time,
            'rank': rank
        }

    except Exception as e:
        print(f"[GPU {rank}] ✗ Failed to generate: {str(e)}")
        return {
            'success': False,
            'index': sample['index'],
            'caption': sample['caption'],
            'error': str(e),
            'rank': rank
        }

def generate_report(results, output_dir, total_time, rank):
    """Generate a summary report of the inference results for this GPU."""
    report_path = output_dir / f"phygenbench_inference_report_gpu{rank}.md"

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    with open(report_path, "w") as f:
        f.write(f"# PhyGenBench Wan2.2-T2V-5B Inference Report - GPU {rank}\n\n")
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
            f.write(f"- [{r['index']:04d}] {r['caption'][:60]}... ({r['category']}) - {r['generation_time']:.2f}s\n")

        if failed:
            f.write("\n## Failed Generations\n")
            for r in failed:
                f.write(f"- [{r['index']:04d}] {r['caption'][:60]}... - Error: {r['error']}\n")

    print(f"[GPU {rank}] 📊 Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU PhyGenBench Wan2.2-T2V-5B Inference Pipeline")
    parser.add_argument("--rank", type=int, required=True,
                       help="GPU rank (0, 1, 2, or 3 for 4 GPUs)")
    parser.add_argument("--world_size", type=int, default=4,
                       help="Total number of GPUs to use")
    parser.add_argument("--prompts_json", type=str, default="phygenbench_prompts.json",
                       help="Path to PhyGenBench prompts JSON file")
    parser.add_argument("--output_dir", type=str, default="phygenbench_results",
                       help="Output directory for generated videos")
    parser.add_argument("--model_name", type=str, default="wan22_t2v_5b",
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
                       help="Output video FPS")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set device for this rank
    device = f"cuda:{args.rank}"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print(f"PhyGenBench Wan2.2-T2V-5B Inference Pipeline - GPU {args.rank}")
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
        'negative_prompt': "blurry, low quality, distorted, static image, cartoon, animation"
    }

    print(f"\n[GPU {args.rank}] Configuration:")
    for key, value in video_config.items():
        print(f"  {key}: {value}")

    # Load data
    print(f"\n[GPU {args.rank}] Loading prompts from: {args.prompts_json}")

    all_data = load_prompts(args.prompts_json, args.max_samples)
    print(f"\n[GPU {args.rank}] Total prompts available: {len(all_data)}")

    # Shard data for this GPU
    data = shard_data(all_data, args.rank, args.world_size)

    if not data:
        print(f"[GPU {args.rank}] No data assigned to this GPU!")
        return

    # Load model on the assigned GPU
    pipe = load_wan22_pipeline(device=device, use_t2v=True)

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
