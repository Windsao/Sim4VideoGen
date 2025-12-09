#!/bin/bash

# Training script for WAN model with motion vector prediction
#
# This script trains the WAN model to jointly predict:
# 1. Video frames (standard diffusion denoising)
# 2. Motion vectors (physics-aware motion prediction)
#
# The dataset should have the following structure:
#   base_dir/
#       test_scenario_1/
#           env_0/0/0/
#               rgb/
#                   rgb_0001.png, rgb_0002.png, ...
#               motion_vectors/
#                   motion_vectors_0001.npy, motion_vectors_0002.npy, ...
#       test_scenario_2/
#           ...

# ============================================
# Configuration
# ============================================

# Parse arguments - support both positional and flexible ordering
# Usage: ./train_wan_with_motion.sh [source_dir] [training_mode]
#    or: ./train_wan_with_motion.sh [training_mode]  (uses default source_dir)
#
# Training modes:
#   lora        - LoRA + motion head + depth head (full adaptation)
#   motion_last - motion head + DiT last layer only
#   depth_last  - depth head + DiT last layer only
#   both_last   - motion head + depth head + DiT last layer
#   motion_only - motion head only (no video improvement)
#   depth_only  - depth head only (no video improvement)
SOURCE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
TRAINING_MODE="lora"

for arg in "$@"; do
    case "$arg" in
        lora|motion_last|depth_last|both_last|motion_only|depth_only)
            TRAINING_MODE="$arg"
            ;;
        *)
            # Assume it's a path if it's not a known mode
            SOURCE_DIR="$arg"
            ;;
    esac
done

OUTPUT_METADATA_DIR="data/sim_physics_motion_dataset"
MODEL_BASE_PATH="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI"

# Set output model path based on training mode
case "${TRAINING_MODE}" in
    "lora")
        OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_lora"
        ;;
    "motion_last")
        OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_motion_last"
        ;;
    "depth_last")
        OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_depth_last"
        ;;
    "both_last")
        OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_both_last"
        ;;
    "motion_only")
        OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_motion_only"
        ;;
    "depth_only")
        OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_depth_only"
        ;;
esac

# Training hyperparameters
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
DATASET_REPEAT=100
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LORA_RANK=32
GRADIENT_ACCUMULATION_STEPS=4

# Motion-specific parameters
MOTION_CHANNELS=4
MOTION_LOSS_WEIGHT=0.1
MOTION_LOSS_TYPE="mse"
MOTION_SCALE=0.01  # Scale down large motion values

# Depth-specific parameters
DEPTH_LOSS_WEIGHT=0.1
DEPTH_LOSS_TYPE="mse"
DEPTH_SCALE=1.0  # Scale for depth values (distance to camera in meters)

# Model paths
MODEL_PATHS="[\"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors\", \"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth\", \"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth\"]"

echo "========================================="
echo "WAN Training with Motion + Depth Loss"
echo "========================================="
echo ""
echo "Source directory: ${SOURCE_DIR}"
echo "Output model path: ${OUTPUT_MODEL_PATH}"
echo "Training mode: ${TRAINING_MODE}"
echo ""
echo "Motion settings:"
echo "  - Motion loss weight: ${MOTION_LOSS_WEIGHT}"
echo "  - Motion channels: ${MOTION_CHANNELS}"
echo "  - Motion scale: ${MOTION_SCALE}"
echo ""
echo "Depth settings:"
echo "  - Depth loss weight: ${DEPTH_LOSS_WEIGHT}"
echo "  - Depth scale: ${DEPTH_SCALE}"
echo ""

# Set training mode flags
LORA_ARGS=""
wandb_run_name=""
case "${TRAINING_MODE}" in
    "lora")
        LORA_ARGS="--lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 --lora_rank ${LORA_RANK}"
        echo "Mode: LoRA + Motion Head + Depth Head (full adaptation)"
        wandb_run_name="lora"
        ;;
    "motion_last")
        echo "Mode: Motion Head + DiT Last Layer"
        wandb_run_name="motion_last"
        ;;
    "depth_last")
        echo "Mode: Depth Head + DiT Last Layer"
        wandb_run_name="depth_last"
        ;;
    "both_last")
        echo "Mode: Motion Head + Depth Head + DiT Last Layer"
        wandb_run_name="both_last"
        ;;
    "motion_only")
        echo "Mode: Motion Head only (no video improvement)"
        wandb_run_name="motion_only"
        ;;
    "depth_only")
        echo "Mode: Depth Head only (no video improvement)"
        wandb_run_name="depth_only"
        ;;
    *)
        echo "Error: Unknown training mode '${TRAINING_MODE}'"
        echo "Valid modes: lora, motion_last, depth_last, both_last, motion_only, depth_only"
        exit 1
        ;;
esac
echo ""

# ============================================
# Step 1: Prepare the dataset metadata
# ============================================
echo "Step 1/2: Preparing dataset metadata..."
echo "Scanning all test scenarios in: ${SOURCE_DIR}"

# Create metadata directory
mkdir -p "${OUTPUT_METADATA_DIR}"

# Generate metadata CSV for directories with motion vectors
python -c "
import os
import csv
import glob

source_dir = '${SOURCE_DIR}'
output_dir = '${OUTPUT_METADATA_DIR}'

# Find all directories containing both rgb and motion_vectors
entries = []

for root, dirs, files in os.walk(source_dir):
    # Check if this directory has both rgb and motion_vectors subdirs
    rgb_dir = os.path.join(root, 'rgb')
    motion_dir = os.path.join(root, 'motion_vectors')

    if os.path.isdir(rgb_dir) and os.path.isdir(motion_dir):
        # Check that both have files
        rgb_files = glob.glob(os.path.join(rgb_dir, '*.png'))
        motion_files = glob.glob(os.path.join(motion_dir, '*.npy'))

        if len(rgb_files) >= 10 and len(motion_files) >= 10:
            # Get relative path from source_dir
            rel_path = os.path.relpath(root, source_dir)

            # Extract scenario name for prompt
            parts = rel_path.split(os.sep)
            scenario = parts[0] if parts else 'physics simulation'
            scenario = scenario.replace('_', ' ').replace('test ', '')

            prompt = f'A physics simulation showing {scenario}'

            entries.append({
                'video': rel_path,
                'prompt': prompt,
                'num_frames': len(rgb_files),
                'num_motion': len(motion_files),
            })

print(f'Found {len(entries)} valid sequences with motion vectors')

# Write metadata CSV
csv_path = os.path.join(output_dir, 'metadata.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video', 'prompt', 'num_frames', 'num_motion'])
    writer.writeheader()
    writer.writerows(entries)

print(f'Metadata written to: {csv_path}')
"

if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed!"
    exit 1
fi

echo ""
echo "Dataset metadata prepared successfully!"
echo ""

# ============================================
# Step 2: Run training with motion + depth loss
# ============================================
echo "Step 2/2: Starting training with motion vector and depth loss..."
echo ""

accelerate launch train_wan_with_motion.py \
    --dataset_base_path "${SOURCE_DIR}" \
    --dataset_metadata_path "${OUTPUT_METADATA_DIR}/metadata.csv" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --dataset_repeat ${DATASET_REPEAT} \
    --model_paths "${MODEL_PATHS}" \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${OUTPUT_MODEL_PATH}" \
    ${LORA_ARGS} \
    --save_steps 100 \
    --use_gradient_checkpointing_offload \
    --training_mode ${TRAINING_MODE} \
    --motion_channels ${MOTION_CHANNELS} \
    --motion_loss_weight ${MOTION_LOSS_WEIGHT} \
    --motion_loss_type ${MOTION_LOSS_TYPE} \
    --motion_scale ${MOTION_SCALE} \
    --depth_loss_weight ${DEPTH_LOSS_WEIGHT} \
    --depth_loss_type ${DEPTH_LOSS_TYPE} \
    --depth_scale ${DEPTH_SCALE} \
    --use_wandb \
    --wandb_project "Sim4Videos" \
    --wandb_run_name ${wandb_run_name} \

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
echo "Model saved to: ${OUTPUT_MODEL_PATH}"
echo ""
echo "To run inference with motion output:"
echo "  python inference_wan_with_motion.py \\"
echo "      --lora_checkpoint ${OUTPUT_MODEL_PATH}/step-XXXX.safetensors \\"
echo "      --prompt 'Your prompt here' \\"
echo "      --output output.mp4 \\"
echo "      --output_motion motion.npy"
echo ""
