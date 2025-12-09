#!/usr/bin/env python3
"""
Script to prepare image sequence dataset for WAN video finetuning.

This script scans a directory containing image sequences organized in subdirectories
and creates a metadata CSV file for training.

Expected data structure:
    base_dir/
        env_0/0/0/rgb/
            rgb_0001.png
            rgb_0002.png
            ...
        env_1/0/0/rgb/
            ...

Each rgb directory is treated as one video sample.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def find_image_sequence_dirs(base_dir, pattern="rgb"):
    """
    Find all directories containing image sequences.

    Args:
        base_dir: Root directory to search
        pattern: Directory name pattern to match (default: "rgb")

    Returns:
        List of paths to directories containing image sequences
    """
    image_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if pattern in os.path.basename(root):
            # Check if directory contains images
            has_images = any(
                f.endswith(('.png', '.jpg', '.jpeg'))
                for f in os.listdir(root)
            )
            if has_images:
                image_dirs.append(root)

    return sorted(image_dirs)


def count_images_in_dir(dir_path):
    """Count number of image files in directory."""
    image_extensions = ('.png', '.jpg', '.jpeg')
    return sum(
        1 for f in os.listdir(dir_path)
        if f.lower().endswith(image_extensions)
    )


def generate_prompt_from_path(path, base_dir):
    """
    Generate a descriptive prompt based on the path structure.

    For Sim_Physics dataset structure: TestOutput/test_xxx/env_0/0/0/rgb
    This extracts the test name and creates a natural language description.

    Args:
        path: Path to the image sequence directory
        base_dir: Base directory of the dataset

    Returns:
        Generated prompt string
    """
    # Extract relative path components
    rel_path = os.path.relpath(path, base_dir)
    parts = Path(rel_path).parts

    # For structure like "test_ball_and_block_fall/env_0/0/0/rgb"
    # Extract the test scenario name
    if len(parts) > 0:
        test_name = parts[0]

        # Remove "test_" prefix if present
        if test_name.startswith("test_"):
            test_name = test_name[5:]

        # Convert underscores to spaces for natural language
        scenario = test_name.replace('_', ' ')

        # Create more descriptive prompts based on common physics scenarios
        prompt_templates = {
            'ball and block fall': 'A physics simulation of a ball and block falling',
            'ball collide': 'A physics simulation showing balls colliding',
            'ball hits duck': 'A physics simulation of a ball hitting a duck',
            'ball hits nothing': 'A physics simulation of a ball moving with no collision',
            'ball in basket': 'A physics simulation of a ball going into a basket',
            'ball ramp': 'A physics simulation of a ball rolling down a ramp',
            'ball rolls off': 'A physics simulation of a ball rolling off a surface',
            'ball rolls on glass': 'A physics simulation of a ball rolling on glass',
            'ball train': 'A physics simulation of multiple balls in sequence',
            'block domino': 'A physics simulation of blocks falling like dominoes',
            'domino in juice': 'A physics simulation of dominoes falling near liquid',
            'domino with space': 'A physics simulation of spaced domino pieces',
            'duck and domino': 'A physics simulation of a duck and domino interaction',
            'duck falls in box': 'A physics simulation of a duck falling into a box',
            'duck static': 'A physics simulation of a static duck object',
            'light on block': 'A physics simulation showing lighting effects on a block',
            'light on mug': 'A physics simulation showing lighting effects on a mug',
            'light on mug block': 'A physics simulation showing lighting on a mug and block',
            'light on sculpture': 'A physics simulation showing lighting on a sculpture',
            'roll behind box': 'A physics simulation of an object rolling behind a box',
            'roll front box': 'A physics simulation of an object rolling in front of a box',
        }

        # Use specific template if available, otherwise generic
        prompt = prompt_templates.get(scenario, f"A physics simulation showing {scenario}")

        return prompt
    else:
        return "A physics simulation video"


def prepare_image_sequence_dataset(
    source_dir,
    output_dir,
    pattern="rgb",
    custom_prompts=None,
    default_prompt=None
):
    """
    Prepare the dataset for WAN video finetuning from image sequences.

    Args:
        source_dir: Path to the dataset root directory
        output_dir: Directory to save the prepared metadata
        pattern: Directory name pattern to match (default: "rgb")
        custom_prompts: Optional dict mapping paths to custom prompts
        default_prompt: Default prompt if not generated from path

    Returns:
        True if successful, False otherwise
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Check if source directory exists
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_path}")
        return False

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Searching for image sequences in: {source_path}")
    print(f"Looking for directories matching pattern: '{pattern}'")

    # Find all image sequence directories
    image_dirs = find_image_sequence_dirs(str(source_path), pattern)

    if not image_dirs:
        print(f"Error: No image sequence directories found in {source_path}")
        return False

    print(f"\nFound {len(image_dirs)} image sequence directories")

    # Prepare metadata
    metadata = []

    for dir_path in tqdm(image_dirs, desc="Processing image sequences"):
        # Count images in this directory
        num_images = count_images_in_dir(dir_path)

        if num_images == 0:
            print(f"Warning: Skipping empty directory: {dir_path}")
            continue

        # Generate or use custom prompt
        if custom_prompts and dir_path in custom_prompts:
            prompt = custom_prompts[dir_path]
        elif default_prompt:
            prompt = default_prompt
        else:
            prompt = generate_prompt_from_path(dir_path, str(source_path))

        # Add to metadata - store absolute path to the directory
        metadata.append({
            'video': os.path.abspath(dir_path),
            'prompt': prompt,
            'negative_prompt': "",  # Can be customized
            'num_images': num_images
        })

    if not metadata:
        print("Error: No valid image sequences found")
        return False

    # Create metadata CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = output_path / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"\nDataset prepared successfully!")
    print(f"   - Metadata saved to: {metadata_csv_path}")
    print(f"   - Total samples: {len(metadata)}")
    print(f"   - Images per sample: min={metadata_df['num_images'].min()}, "
          f"max={metadata_df['num_images'].max()}, "
          f"mean={metadata_df['num_images'].mean():.1f}")

    # Print first few entries
    print(f"\nFirst 3 metadata entries:")
    for i, entry in enumerate(metadata[:3]):
        print(f"   {i+1}. Video dir: {entry['video']}")
        print(f"      Prompt: {entry['prompt']}")
        print(f"      Images: {entry['num_images']}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare image sequence dataset for WAN video training"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source directory containing image sequences"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sim_physics_dataset",
        help="Directory to save the metadata CSV (default: data/sim_physics_dataset)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="rgb",
        help="Directory name pattern to match (default: 'rgb')"
    )
    parser.add_argument(
        "--default_prompt",
        type=str,
        default=None,
        help="Default prompt to use for all samples (optional)"
    )

    args = parser.parse_args()

    success = prepare_image_sequence_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        default_prompt=args.default_prompt
    )

    if not success:
        print("\nDataset preparation failed. Please check the error messages above.")
        exit(1)
    else:
        print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
