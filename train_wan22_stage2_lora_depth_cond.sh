#!/bin/bash

###############################################################################
# Stage 2: LoRA + Motion/Depth Heads with Depth-Image Conditioning
#
# This script mirrors train_wan22_stage2_lora.sh but enables depth-image
# conditioning by passing --extra_inputs "input_image,depth_image".
#
# It expects depth maps under distance_to_camera/ and will use the first
# frame depth map as the conditioning depth_image.
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

# Output path
OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage2_depth_cond"

# Video dimensions (must match Stage 1)
HEIGHT=480
WIDTH=480
NUM_FRAMES=49

# ============================================
# Model Configuration (Local Paths)
# ============================================

WAN22_MODEL_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B"
WAN21_T5_MODEL="${WAN22_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
WAN21_TOKENIZER_DIR="${WAN22_MODEL_DIR}/google/umt5-xxl"

WAN22_DIT_DIR="${WAN22_MODEL_DIR}"
if [ -d "${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B" ]; then
  WAN22_DIT_DIR="${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B"
fi

# Load Stage 1 checkpoints
STAGE1_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1/final"
MOTION_HEAD_CHECKPOINT="${STAGE1_PATH}/motion_head.pth"
DEPTH_HEAD_CHECKPOINT="${STAGE1_PATH}/depth_head.pth"

# Check if Stage 1 checkpoints exist
if [ ! -f "$MOTION_HEAD_CHECKPOINT" ]; then
    echo "ERROR: Motion head checkpoint not found at $MOTION_HEAD_CHECKPOINT"
    echo "Please run Stage 1 training first: bash train_wan22_stage1_heads.sh"
    exit 1
fi

if [ ! -f "$DEPTH_HEAD_CHECKPOINT" ]; then
    echo "ERROR: Depth head checkpoint not found at $DEPTH_HEAD_CHECKPOINT"
    echo "Please run Stage 1 training first: bash train_wan22_stage1_heads.sh"
    exit 1
fi

echo "=========================================================================="
echo "Stage 2: LoRA + Depth Conditioning (Depth Image)"
echo "=========================================================================="
echo "Motion head: $MOTION_HEAD_CHECKPOINT"
echo "Depth head:  $DEPTH_HEAD_CHECKPOINT"
echo "=========================================================================="

# LoRA configuration
LORA_RANK=32
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# Training hyperparameters
LEARNING_RATE=1e-5
NUM_EPOCHS=5
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=500

# Loss weights
MOTION_LOSS_WEIGHT=0.5
DEPTH_LOSS_WEIGHT=0.5

# Wandb configuration (optional)
USE_WANDB=true
WANDB_PROJECT="wan22-ti2v-stage2-lora-depth-cond"

###############################################################################
# Run Stage 2 Training
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
  --training_mode lora \
  --extra_inputs "input_image,depth_image" \
  --motion_channels 4 \
  --lora_rank $LORA_RANK \
  --lora_target_modules "$LORA_TARGET_MODULES" \
  --motion_head_checkpoint "$MOTION_HEAD_CHECKPOINT" \
  --depth_head_checkpoint "$DEPTH_HEAD_CHECKPOINT" \
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
echo "Stage 2 Training Complete!"
echo "=========================================================================="
echo "Final model has been saved to:"
echo "  $OUTPUT_PATH/final/motion_head.pth"
echo "  $OUTPUT_PATH/final/depth_head.pth"
echo "  $OUTPUT_PATH/final/lora_weights.pth"
