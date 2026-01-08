#!/bin/bash

###############################################################################
# Stage 1: Train Motion/Depth Heads with Full-Resolution Outputs
#
# This stage trains motion/depth heads that upsample to full video resolution.
# Checkpoints will be saved to:
#   ${MODEL_BASE_PATH}/wan22_ti2v_stage1_fullres/checkpoint-{step}/
#   ${MODEL_BASE_PATH}/wan22_ti2v_stage1_fullres/final/
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
USE_LARGE_DATASET=true

if [ "$USE_LARGE_DATASET" = true ]; then
  DATASET_PRESET="magic_physics"
  DATASET_BASE_PATH="/nyx-storage1/hanliu/MagicPhysics"
  DATASET_METADATA_PATH="/home/mzh1800/DiffSynth-Studio/data/magic_physics_dataset/metadata.csv"
else
  DATASET_PRESET="sim_physics"
fi

# ============================================
# Video Configuration
# ============================================

# Video dimensions (matched to Sim_Physics dataset)
HEIGHT=480
WIDTH=480
NUM_FRAMES=49

# ============================================
# Model Configuration (Local Paths)
# ============================================

WAN22_MODEL_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B"
WAN21_T5_MODEL="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
WAN21_TOKENIZER_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B/google/umt5-xxl"

# Prefer nested DiT folder when available.
WAN22_DIT_DIR="${WAN22_MODEL_DIR}"
if [ -d "${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B" ]; then
  WAN22_DIT_DIR="${WAN22_MODEL_DIR}/Wan-AI/Wan2___2-TI2V-5B"
fi

# ============================================
# Training Hyperparameters
# ============================================

LEARNING_RATE=1e-4
NUM_EPOCHS=10
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=500

# ============================================
# Loss Weights
# ============================================

MOTION_LOSS_WEIGHT=1.0
DEPTH_LOSS_WEIGHT=1.0
USE_WARP_LOSS=true
WARP_LOSS_WEIGHT=0.1
WARP_LOSS_TYPE="mse"

# Optional: Use spatio-temporal depth head (keep in sync with Stage 2)
USE_SPATIOTEMPORAL_DEPTH=true
SPATIOTEMPORAL_DEPTH_TYPE="full"  # "simple" or "full"

# Optional: Use spatio-temporal motion head (keep in sync with Stage 2)
USE_SPATIOTEMPORAL_MOTION=true
SPATIOTEMPORAL_MOTION_TYPE="full"  # "simple" or "full"

# Output path
if [ "$USE_LARGE_DATASET" = true ]; then
  OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1_fullres_large"
else
  OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage1_fullres"
fi

# ============================================
# Wandb Configuration (Optional)
# ============================================

USE_WANDB=true
if [ "$USE_LARGE_DATASET" = true ]; then
  WANDB_PROJECT="wan22-ti2v-stage1-heads-fullres-large"
  WANDB_NAME="wan22-ti2v-stage1-heads-fullres-large"
else
  WANDB_PROJECT="wan22-ti2v-stage1-heads-fullres"
  WANDB_NAME="wan22-ti2v-stage1-heads-fullres"
fi

# ============================================
# Print Configuration
# ============================================

echo "========================================="
echo "WAN2.2-5B Stage 1: Train Full-Res Heads"
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
echo "  - Warp loss: ${USE_WARP_LOSS} (weight ${WARP_LOSS_WEIGHT}, type ${WARP_LOSS_TYPE})"
echo ""

###############################################################################
# Run Stage 1 Training
###############################################################################

accelerate launch --mixed_precision bf16 --num_processes 4 \
  train_wan22_ti2v_motion_depth.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --dataset_preset "$DATASET_PRESET" \
  --dataset_repeat 5 \
  --validate_dataset_paths \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --model_paths "[\"${WAN22_DIT_DIR}\", \"${WAN21_T5_MODEL}\", \"${WAN22_MODEL_DIR}/Wan2.2_VAE.pth\"]" \
  --tokenizer_path "${WAN21_TOKENIZER_DIR}" \
  --output_path "$OUTPUT_PATH" \
  --training_mode heads_only \
  --full_res_heads \
  --motion_channels 4 \
  --motion_loss_weight $MOTION_LOSS_WEIGHT \
  --depth_loss_weight $DEPTH_LOSS_WEIGHT \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --save_steps $SAVE_STEPS \
  $([ "$USE_WARP_LOSS" = true ] && echo "--use_warp_loss --warp_loss_weight $WARP_LOSS_WEIGHT --warp_loss_type $WARP_LOSS_TYPE" || echo "") \
  $([ "$USE_SPATIOTEMPORAL_DEPTH" = true ] && echo "--use_spatiotemporal_depth --spatiotemporal_depth_type $SPATIOTEMPORAL_DEPTH_TYPE" || echo "") \
  $([ "$USE_SPATIOTEMPORAL_MOTION" = true ] && echo "--use_spatiotemporal_motion --spatiotemporal_motion_type $SPATIOTEMPORAL_MOTION_TYPE" || echo "") \
  $([ "$USE_WANDB" = true ] && echo "--use_wandb --wandb_project $WANDB_PROJECT --wandb_name $WANDB_NAME" || echo "")

echo ""
echo "=========================================================================="
echo "Stage 1 Full-Res Training Complete!"
echo "=========================================================================="
echo "Motion and depth heads saved to:"
echo "  $OUTPUT_PATH/final/motion_head.pth"
echo "  $OUTPUT_PATH/final/depth_head.pth"
echo "=========================================================================="
