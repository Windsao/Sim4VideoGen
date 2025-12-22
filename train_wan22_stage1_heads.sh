#!/bin/bash

###############################################################################
# Stage 1: Train Motion Head and Depth Head Only (Backbone Frozen)
#
# This stage trains only the motion and depth prediction heads while keeping
# the DiT backbone completely frozen. This allows the heads to learn their
# tasks without interfering with the pre-trained video generation model.
#
# After training completes, checkpoints will be saved to:
#   ${MODEL_BASE_PATH}/wan22_ti2v_stage1/checkpoint-{step}/
#   ${MODEL_BASE_PATH}/wan22_ti2v_stage1/final/
###############################################################################

# Set CUDA device (modify if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================
# Paths Configuration
# ============================================

# Base path for models (modify to your local path)
MODEL_BASE_PATH="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI"

# Training data paths
DATASET_BASE_PATH="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
DATASET_METADATA_PATH="/home/mzh1800/DiffSynth-Studio/data/sim_physics_metadata.csv"

# Output path (will save checkpoints locally)
OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1"

# ============================================
# Video Configuration
# ============================================

# Video dimensions (matched to Sim_Physics dataset)
HEIGHT=480
WIDTH=480
NUM_FRAMES=49  # Will sample 49 frames from 131 available frames

# ============================================
# Model Configuration (Local Paths)
# ============================================

# Use locally downloaded WAN2.2-5B model checkpoints
# Note: Wan2.2-TI2V-5B uses sharded model format
# - DiT: Point to directory (auto-loads all sharded safetensors files)
# - T5: Use from Wan2.1 (shared across versions)
# - VAE: Use from Wan2.2
WAN22_MODEL_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B"
WAN21_T5_MODEL="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
WAN21_TOKENIZER_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B/google/umt5-xxl"

# Some WAN2.2 releases store the sharded DiT weights in a nested folder with an index JSON.
# Prefer that folder when it exists.
WAN22_DIT_DIR="${WAN22_MODEL_DIR}"
if [ -d "${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B" ]; then
  WAN22_DIT_DIR="${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B"
fi

# ============================================
# Training Hyperparameters
# ============================================

LEARNING_RATE=1e-4  # Higher LR for heads-only training
NUM_EPOCHS=10
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=500

# ============================================
# Loss Weights
# ============================================

MOTION_LOSS_WEIGHT=1.0
DEPTH_LOSS_WEIGHT=1.0

# ============================================
# Wandb Configuration (Optional)
# ============================================

USE_WANDB=true
WANDB_PROJECT="wan22-ti2v-stage1-heads"

# ============================================
# Print Configuration
# ============================================

echo "========================================="
echo "WAN2.2-5B Stage 1: Train Heads Only"
echo "========================================="
echo ""
echo "Model base path: ${MODEL_BASE_PATH}"
echo "Dataset path: ${DATASET_BASE_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo ""
echo "Video settings:"
echo "  - Height: ${HEIGHT}"
echo "  - Width: ${WIDTH}"
echo "  - Num frames: ${NUM_FRAMES}"
echo ""
echo "Training settings:"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Num epochs: ${NUM_EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo ""
echo "Loss weights:"
echo "  - Motion loss: ${MOTION_LOSS_WEIGHT}"
echo "  - Depth loss: ${DEPTH_LOSS_WEIGHT}"
echo ""

###############################################################################
# Run Stage 1 Training
###############################################################################

accelerate launch --mixed_precision bf16 --num_processes 4 \
  train_wan22_ti2v_motion_depth.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --dataset_repeat 100 \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --model_paths "[\"${WAN22_DIT_DIR}\", \"${WAN21_T5_MODEL}\", \"${WAN22_MODEL_DIR}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "${WAN21_TOKENIZER_DIR}" \
  --output_path "$OUTPUT_PATH" \
  --training_mode heads_only \
  --motion_channels 4 \
  --motion_loss_weight $MOTION_LOSS_WEIGHT \
  --depth_loss_weight $DEPTH_LOSS_WEIGHT \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --save_steps $SAVE_STEPS \
  $([ "$USE_WANDB" = true ] && echo "--use_wandb --wandb_project $WANDB_PROJECT" || echo "")

echo ""
echo "=========================================================================="
echo "Stage 1 Training Complete!"
echo "=========================================================================="
echo "Motion and depth heads have been trained and saved to:"
echo "  $OUTPUT_PATH/final/motion_head.pth"
echo "  $OUTPUT_PATH/final/depth_head.pth"
echo ""
echo "Next step: Run stage 2 to fine-tune the backbone with LoRA:"
echo "  bash train_wan22_stage2_lora.sh"
echo "=========================================================================="
