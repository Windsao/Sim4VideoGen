#!/bin/bash

# Script to train WAN model on Sim_Physics image sequence dataset
# This script uses LoRA fine-tuning for the Wan2.2 model with image sequences
#
# This will scan ALL test scenarios under TestOutput directory:
# - test_ball_and_block_fall, test_ball_collide, test_ball_hits_duck, etc.
# Each test scenario is automatically included in training

# Configuration
SOURCE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
OUTPUT_METADATA_DIR="data/sim_physics_dataset"
MODEL_BASE_PATH="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI"
OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.2_lora"

# Training hyperparameters
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
DATASET_REPEAT=100
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LORA_RANK=32
GRADIENT_ACCUMULATION_STEPS=4

# Model paths - Update these for WAN2.2
# Note: You may need to adjust these paths based on your WAN2.2 model location
MODEL_PATHS="[\"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors\", \"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth\", \"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth\"]"

echo "========================================="
echo "WAN2.2 Fine-tuning on Image Sequences"
echo "========================================="
echo ""
echo "Source directory: ${SOURCE_DIR}"
echo "Output model path: ${OUTPUT_MODEL_PATH}"
echo ""

# Step 1: Prepare the dataset metadata
echo "Step 1/2: Preparing dataset metadata..."
echo "Scanning all test scenarios in: ${SOURCE_DIR}"
python prepare_image_dataset.py \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_METADATA_DIR}" \
  --pattern "rgb"

if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed!"
    exit 1
fi

echo ""
echo "Dataset metadata prepared successfully!"
echo ""

# Step 2: Run LoRA training
echo "Step 2/2: Starting LoRA training for WAN2.2..."
echo ""

accelerate launch train_wan_image_sequences.py \
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
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank ${LORA_RANK} \
  --save_steps 100 \
  --use_gradient_checkpointing_offload \
  --image_pattern "*.png"

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
