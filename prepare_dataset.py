#!/usr/bin/env python3
"""
Script to prepare the fold_towel_new dataset from local downloaded files for WAN video finetuning.
This script uses locally downloaded videos and creates the required metadata CSV file.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_wan_dataset_from_local(source_dir="/nyx-storage1/hanliu/fold_towel_new", output_dir="data/fold_towel_dataset", use_symlinks=False):
    """
    Prepare the dataset for WAN video finetuning from locally downloaded videos.

    Args:
        source_dir: Path to the locally downloaded dataset with videos folder
        output_dir: Directory to save the prepared dataset
        use_symlinks: If True, create symlinks instead of copying videos
    """

    # Convert to Path objects
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Check if source directory exists
    if not source_path.exists():
        print(f"❌ Error: Source directory not found: {source_path}")
        return False

    # Check if videos folder exists in source
    source_videos_path = source_path / "videos"
    if not source_videos_path.exists():
        print(f"❌ Error: Videos folder not found in: {source_path}")
        return False

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Using local dataset from: {source_path}")
    print(f"📂 Videos source folder: {source_videos_path}")

    # Get all video files from source (including subdirectories)
    video_files = sorted(list(source_videos_path.glob("**/*.mp4")))

    if not video_files:
        print(f"❌ Error: No MP4 video files found in {source_videos_path}")
        return False

    print(f"\n📹 Found {len(video_files)} video files")

    # Prepare metadata for CSV
    metadata = []

    # Process each video file
    for idx, source_video_path in enumerate(tqdm(video_files, desc="Processing videos")):
        # Use absolute path to source video directly
        video_abs_path = source_video_path.resolve()

        # Generate prompt based on video filename or index
        # You can customize this based on your specific dataset structure
        # For now, using a generic prompt about towel folding
        prompt = f"A video showing the process of folding a towel, demonstration {idx+1}"

        # If there's a metadata file or text file with prompts in the source directory,
        # you can load them here instead of using generic prompts

        # Add to metadata - use absolute path to source video
        metadata.append({
            'video': str(video_abs_path),
            'prompt': prompt,
            'negative_prompt': ""  # You can customize this if needed
        })

    # Check if there's an existing metadata file in the source directory
    # that we can use for prompts
    source_metadata_files = list(source_path.glob("*.csv")) + list(source_path.glob("*.json"))
    if source_metadata_files:
        print(f"\n📄 Found potential metadata files in source: {[f.name for f in source_metadata_files]}")
        print("   Consider loading prompts from these files if they contain text descriptions")

    # Create metadata CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_csv_path = output_path / "metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"\n✅ Dataset prepared successfully!")
    print(f"   - Videos located at: {source_videos_path}")
    print(f"   - Metadata saved to: {metadata_csv_path}")
    print(f"   - Total samples: {len(metadata)}")

    # Print first few entries of metadata for verification
    print(f"\n📋 First 3 metadata entries:")
    for i, entry in enumerate(metadata[:3]):
        print(f"   {i+1}. Video: {entry['video']}, Prompt: {entry['prompt'][:50]}...")

    # Print sample training command
    print(f"\n📝 Sample training command:")
    print(f"""
accelerate launch examples/wanvideo/model_training/train.py \\
  --dataset_base_path {output_dir} \\
  --dataset_metadata_path {output_dir}/metadata.csv \\
  --height 480 \\
  --width 832 \\
  --dataset_repeat 100 \\
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \\
  --learning_rate 1e-4 \\
  --num_epochs 5 \\
  --remove_prefix_in_ckpt "pipe.dit." \\
  --output_path "./models/train/fold_towel_Wan2.1-T2V-1.3B_lora" \\
  --lora_base_model "dit" \\
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \\
  --lora_rank 32
    """)

    return True

if __name__ == "__main__":
    # You can customize these parameters
    success = prepare_wan_dataset_from_local(
        source_dir="/nyx-storage1/hanliu/fold_towel_new",
        output_dir="data/fold_towel_dataset"
    )

    if not success:
        print("\n⚠️ Dataset preparation encountered issues. Please check the error messages above.")