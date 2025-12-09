#!/usr/bin/env python3
"""
Download Wan2.2-TI2V-5B model to local path
This model supports both Text-to-Video and Text+Image-to-Video generation
"""

import os
from modelscope import snapshot_download

# Target directory for the TI2V model
LOCAL_MODEL_PATH = "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.2-TI2V-5B"
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"

print("="*70)
print("Downloading Wan2.2-TI2V-5B Model")
print("="*70)
print(f"Model ID: {MODEL_ID}")
print(f"Destination: {LOCAL_MODEL_PATH}")
print()

# Create the directory if it doesn't exist
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# Download the model
print("Starting download... This may take a while (~20GB total)")
print()
print("Required files:")
print("  - models_t5_umt5-xxl-enc-bf16.pth (~11GB)")
print("  - diffusion_pytorch_model*.safetensors (~7GB total)")
print("  - Wan2.2_VAE.pth (~490MB)")
print()

try:
    model_path = snapshot_download(
        model_id=MODEL_ID,
        cache_dir=LOCAL_MODEL_PATH,
        revision="master"
    )

    print()
    print("="*70)
    print("✅ Download completed successfully!")
    print("="*70)
    print(f"Model downloaded to: {model_path}")
    print()

    # List the downloaded files
    print("Downloaded files:")
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_size_gb = file_size / (1024**3)  # Size in GB
            total_size += file_size
            rel_path = os.path.relpath(file_path, model_path)
            print(f"  {rel_path} ({file_size_gb:.2f} GB)")

    print()
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print()
    print("The model is now ready to use with your inference script!")

except Exception as e:
    print()
    print("="*70)
    print("❌ Download failed!")
    print("="*70)
    print(f"Error: {str(e)}")
    print()
    print("Please check:")
    print("1. Internet connection")
    print("2. ModelScope access")
    print("3. Disk space available")
    exit(1)
