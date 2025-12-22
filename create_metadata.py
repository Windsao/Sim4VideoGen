#!/usr/bin/env python3
"""
Generate metadata CSV for Sim_Physics dataset.
"""

import os
import csv
from pathlib import Path

def find_sample_directories(base_path):
    """Find all directories containing rgb, motion_vectors, and distance_to_camera subdirs."""
    samples = []

    for root, dirs, files in os.walk(base_path):
        # Check if this directory has the required subdirectories
        if 'rgb' in dirs and 'motion_vectors' in dirs and 'distance_to_camera' in dirs:
            # Get relative path from base_path
            rel_path = os.path.relpath(root, base_path)
            samples.append(rel_path)

    return sorted(samples)

def create_metadata_csv(base_path, output_path):
    """Create metadata CSV file."""
    samples = find_sample_directories(base_path)

    print(f"Found {len(samples)} samples")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['video', 'prompt'])

        # Write each sample
        for sample in samples:
            # Extract a description from the path
            parts = sample.split('/')
            scene_name = parts[0].replace('test_', '').replace('_', ' ')
            prompt = f"A physics simulation of {scene_name}"

            writer.writerow([sample, prompt])

    print(f"Created metadata CSV: {output_path}")
    print(f"Sample entries:")
    for i, sample in enumerate(samples[:5]):
        print(f"  {i+1}. {sample}")
    if len(samples) > 5:
        print(f"  ... and {len(samples) - 5} more")

if __name__ == "__main__":
    base_path = "/nyx-storage1/hanliu/Sim_Physics/TestOutput"
    output_path = "/home/mzh1800/DiffSynth-Studio/data/sim_physics_metadata.csv"

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    create_metadata_csv(base_path, output_path)
