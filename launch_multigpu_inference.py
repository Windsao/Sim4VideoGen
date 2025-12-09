#!/usr/bin/env python3
"""
Python launcher for multi-GPU inference
Alternative to the bash launcher script
"""

import subprocess
import sys
import time
from pathlib import Path

def launch_multigpu_inference(
    world_size=4,
    output_dir="physics_iq_results",
    model_name="wan22_ti2v_5b_multigpu",
    max_samples=None,
    height=480,
    width=832,
    num_frames=81,
    cfg_scale=7.0,
    num_inference_steps=50,
    fps=16,
    seed=42,
    resize_input=True
):
    """Launch multiple inference processes, one per GPU."""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Starting Multi-GPU Inference Pipeline")
    print("=" * 60)
    print(f"World size: {world_size} GPUs")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print("=" * 60)
    print()

    processes = []

    # Launch one process per GPU
    for rank in range(world_size):
        print(f"Launching GPU {rank}...")

        # Build command
        cmd = [
            sys.executable,
            "inference_physics_iq_wan22_multigpu.py",
            "--rank", str(rank),
            "--world_size", str(world_size),
            "--output_dir", output_dir,
            "--model_name", model_name,
            "--height", str(height),
            "--width", str(width),
            "--num_frames", str(num_frames),
            "--cfg_scale", str(cfg_scale),
            "--num_inference_steps", str(num_inference_steps),
            "--fps", str(fps),
            "--seed", str(seed),
        ]

        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])

        if resize_input:
            cmd.append("--resize_input")

        # Set environment to use specific GPU
        env = {"CUDA_VISIBLE_DEVICES": str(rank)}

        # Open log file
        log_file = log_dir / f"gpu_{rank}.log"
        log_f = open(log_file, "w")

        # Launch process
        process = subprocess.Popen(
            cmd,
            env={**subprocess.os.environ, **env},
            stdout=log_f,
            stderr=subprocess.STDOUT
        )

        processes.append((process, log_f, rank))
        print(f"GPU {rank} started with PID {process.pid} (log: {log_file})")

        # Small delay to stagger startup
        time.sleep(2)

    print()
    print("=" * 60)
    print("All GPU processes launched!")
    print("=" * 60)
    print()
    print("Monitor progress with:")
    for rank in range(world_size):
        print(f"  tail -f logs/gpu_{rank}.log")
    print()
    print("Waiting for all processes to complete...")
    print("(Press Ctrl+C to interrupt)")
    print()

    # Wait for all processes to complete
    try:
        for process, log_f, rank in processes:
            return_code = process.wait()
            log_f.close()
            if return_code == 0:
                print(f"✓ GPU {rank} completed successfully")
            else:
                print(f"✗ GPU {rank} failed with return code {return_code}")
    except KeyboardInterrupt:
        print("\nInterrupted! Terminating processes...")
        for process, log_f, rank in processes:
            process.terminate()
            log_f.close()
        sys.exit(1)

    print()
    print("=" * 60)
    print("All processes completed!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch multi-GPU inference")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--output_dir", type=str, default="physics_iq_results")
    parser.add_argument("--model_name", type=str, default="wan22_ti2v_5b_multigpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit total samples (for testing)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resize_input", action="store_true", help="Don't resize input images")

    args = parser.parse_args()

    launch_multigpu_inference(
        world_size=args.world_size,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        fps=args.fps,
        seed=args.seed,
        resize_input=not args.no_resize_input
    )
