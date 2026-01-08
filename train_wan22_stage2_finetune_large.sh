#!/bin/bash

###############################################################################
# Stage 2: Full Fine-tune (Large Dataset)
#
# This script uses train_wan22_stage2_finetune.py and loads Stage 1 heads.
# It mirrors the large dataset setup in train_wan22_stage1_heads_fullres.sh.
###############################################################################

set -euo pipefail

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ============================================
# Paths Configuration
# ============================================

# Base path for models (modify to your local path)
MODEL_BASE_PATH="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI"

# Training data paths (large dataset)
USE_LARGE_DATASET=true
DATASET_PRESET="sim_physics"
DATASET_BASE_PATH="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
DATASET_METADATA_PATH="/home/mzh1800/DiffSynth-Studio/data/sim_physics_metadata.csv"

if [ "$USE_LARGE_DATASET" = true ]; then
  DATASET_PRESET="magic_physics"
  DATASET_BASE_PATH="/nyx-storage1/hanliu/MagicPhysics"
  DATASET_METADATA_PATH="/home/mzh1800/DiffSynth-Studio/data/magic_physics_dataset/metadata.csv"
fi

# Video dimensions (must match Stage 1)
HEIGHT=480
WIDTH=480
NUM_FRAMES=49

# ============================================
# Model Configuration (Local Paths)
# ============================================

WAN22_MODEL_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B"
WAN22_T5_MODEL="${WAN22_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
WAN22_TOKENIZER_DIR="${WAN22_MODEL_DIR}/google/umt5-xxl"

# Prefer nested DiT folder when available.
WAN22_DIT_DIR="${WAN22_MODEL_DIR}"
if [ -d "${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B" ]; then
  WAN22_DIT_DIR="${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B"
fi

# Wandb configuration (optional)
USE_WANDB=true
WANDB_PROJECT="wan22-ti2v-stage2-finetune-large"

# Spatio-temporal depth head configuration (must match Stage 1)
USE_SPATIOTEMPORAL_DEPTH=true
SPATIOTEMPORAL_DEPTH_TYPE="full"  # "simple" or "full"

# Spatio-temporal motion head configuration (must match Stage 1)
USE_SPATIOTEMPORAL_MOTION=true
SPATIOTEMPORAL_MOTION_TYPE="full"  # "simple" or "full"

# Stage 1 checkpoints
if [ "$USE_SPATIOTEMPORAL_DEPTH" = true ]; then
  if [ "$SPATIOTEMPORAL_DEPTH_TYPE" = "full" ]; then
    # STAGE1_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1_spatio_depth_full/final"
    STAGE1_PATH="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage1_fullres_large/final"
    WANDB_NAME="wan22-ti2v-stage2-finetune-large-spatio-depth-full"
    OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage2_finetune_large_spatio_depth_full_no_all_uniform"
  else
    STAGE1_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1_spatio_depth_simple/final"
    WANDB_NAME="wan22-ti2v-stage2-finetune-large-spatio-depth-simple"
    OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage2_finetune_large_spatio_depth_simple"
  fi
else
  STAGE1_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1/final"
  WANDB_NAME="wan22-ti2v-stage2-finetune-large"
  OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage2_finetune_large"
fi

MOTION_HEAD_CHECKPOINT="${STAGE1_PATH}/motion_head.pth"
DEPTH_HEAD_CHECKPOINT="${STAGE1_PATH}/depth_head.pth"

if [ ! -f "$MOTION_HEAD_CHECKPOINT" ]; then
  echo "ERROR: Motion head checkpoint not found at $MOTION_HEAD_CHECKPOINT"
  echo "Please run Stage 1 training first."
  exit 1
fi

if [ ! -f "$DEPTH_HEAD_CHECKPOINT" ]; then
  echo "ERROR: Depth head checkpoint not found at $DEPTH_HEAD_CHECKPOINT"
  echo "Please run Stage 1 training first."
  exit 1
fi

# ============================================
# Fine-tune Configuration
# ============================================

# Choose "full" or "custom"
FINETUNE_MODE="custom"

# When FINETUNE_MODE="custom", set regex patterns for trainable params.
TRAINABLE_PARAM_PATTERNS="blocks\\.14\\.|blocks\\.15\\.|blocks\\.16\\.|blocks\\.17\\.|blocks\\.18\\.|blocks\\.19\\.|blocks\\.20\\.|blocks\\.21\\.|blocks\\.22\\.|blocks\\.23\\."
TRAINABLE_SCOPE="dit"  # "dit" or "pipe"

# Select which pipe submodules to save
SAVE_MODULES="dit"
SAVE_TRAINABLE_ONLY=false

# Training hyperparameters
LEARNING_RATE=1e-5
NUM_EPOCHS=10
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=500
TIMESTEP_SAMPLING="uniform"  # "uniform", "high", or "low"
TIMESTEP_SAMPLING_POWER=2.0

# Dataset validation
VALIDATE_DATASET_PATHS=true

# Loss weights
MOTION_LOSS_WEIGHT=0.
DEPTH_LOSS_WEIGHT=0.
USE_WARP_LOSS=false
WARP_LOSS_WEIGHT=0.0
WARP_LOSS_TYPE="mse"
USE_RGB_WARP_LOSS=false
RGB_WARP_LOSS_WEIGHT=0.0
RGB_WARP_LOSS_TYPE="l1"

echo "=========================================================================="
echo "Stage 2 Fine-tuning (Large Dataset, FINETUNE_MODE=${FINETUNE_MODE})"
echo "=========================================================================="
echo "Dataset preset: $DATASET_PRESET"
echo "Dataset base path: $DATASET_BASE_PATH"
echo "Dataset metadata path: $DATASET_METADATA_PATH"
echo "=========================================================================="
echo "Loading Stage 1 checkpoints:"
echo "  Motion head: $MOTION_HEAD_CHECKPOINT"
echo "  Depth head:  $DEPTH_HEAD_CHECKPOINT"
echo "=========================================================================="

accelerate launch --mixed_precision bf16 --num_processes 4 \
  train_wan22_stage2_finetune.py \
  --dataset_preset "$DATASET_PRESET" \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --dataset_repeat 2 \
  $([ "$VALIDATE_DATASET_PATHS" = true ] && echo "--validate_dataset_paths" || echo "") \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --model_paths "[\"${WAN22_DIT_DIR}\", \"${WAN22_T5_MODEL}\", \"${WAN22_MODEL_DIR}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "${WAN22_TOKENIZER_DIR}" \
  --output_path "$OUTPUT_PATH" \
  --finetune_mode "$FINETUNE_MODE" \
  --trainable_param_patterns "$TRAINABLE_PARAM_PATTERNS" \
  --trainable_scope "$TRAINABLE_SCOPE" \
  --save_modules "$SAVE_MODULES" \
  $([ "$SAVE_TRAINABLE_ONLY" = true ] && echo "--save_trainable_only" || echo "") \
  --motion_channels 4 \
  --motion_head_checkpoint "$MOTION_HEAD_CHECKPOINT" \
  --depth_head_checkpoint "$DEPTH_HEAD_CHECKPOINT" \
  --motion_loss_weight $MOTION_LOSS_WEIGHT \
  --depth_loss_weight $DEPTH_LOSS_WEIGHT \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --save_steps $SAVE_STEPS \
  --timestep_sampling "$TIMESTEP_SAMPLING" \
  --timestep_sampling_power $TIMESTEP_SAMPLING_POWER \
  $([ "$USE_WARP_LOSS" = true ] && echo "--use_warp_loss --warp_loss_weight $WARP_LOSS_WEIGHT --warp_loss_type $WARP_LOSS_TYPE" || echo "") \
  $([ "$USE_RGB_WARP_LOSS" = true ] && echo "--use_rgb_warp_loss --rgb_warp_loss_weight $RGB_WARP_LOSS_WEIGHT --rgb_warp_loss_type $RGB_WARP_LOSS_TYPE" || echo "") \
  $([ "$USE_WANDB" = true ] && echo "--use_wandb --wandb_project $WANDB_PROJECT --wandb_name $WANDB_NAME" || echo "") \
  $([ "$USE_SPATIOTEMPORAL_DEPTH" = true ] && echo "--use_spatiotemporal_depth --spatiotemporal_depth_type $SPATIOTEMPORAL_DEPTH_TYPE" || echo "") \
  $([ "$USE_SPATIOTEMPORAL_MOTION" = true ] && echo "--use_spatiotemporal_motion --spatiotemporal_motion_type $SPATIOTEMPORAL_MOTION_TYPE" || echo "")

echo ""
echo "=========================================================================="
echo "Stage 2 Fine-tuning Complete!"
echo "=========================================================================="
echo "Final checkpoints saved to:"
echo "  $OUTPUT_PATH/final/motion_head.pth"
echo "  $OUTPUT_PATH/final/depth_head.pth"
echo "  $OUTPUT_PATH/final/${SAVE_MODULES//,/\.pth and }.pth"
echo "=========================================================================="
