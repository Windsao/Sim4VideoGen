#!/bin/bash

###############################################################################
# Stage 2: Train DiT Backbone (LoRA) + Motion/Depth Heads Together
#
# This stage loads the pre-trained motion and depth heads from Stage 1 and
# continues training with LoRA adaptation on the DiT backbone. This allows
# the entire model to be fine-tuned end-to-end while keeping the training
# efficient through LoRA.
#
# Prerequisites:
#   - Stage 1 must be completed first
#   - Motion head and depth head checkpoints must exist at:
#     ${MODEL_BASE_PATH}/wan22_ti2v_stage1/final/motion_head.pth
#     ${MODEL_BASE_PATH}/wan22_ti2v_stage1/final/depth_head.pth
#
# After training completes, checkpoints will be saved to:
#   ./models/wan22_ti2v_stage2/checkpoint-{step}/
#   ./models/wan22_ti2v_stage2/final/
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
OUTPUT_PATH="${MODEL_BASE_PATH}/wan22_ti2v_stage2"

# Video dimensions (must match Stage 1)
HEIGHT=480
WIDTH=480
NUM_FRAMES=49

# ============================================
# Model Configuration (Local Paths)
# ============================================

# Use locally downloaded WAN2.2-5B model checkpoints
# Note: Wan2.2-TI2V-5B uses sharded model format
# - DiT: Point to directory (auto-loads all sharded safetensors files)
# - T5: Use from Wan2.1 (shared across versions)
# - VAE: Use from Wan2.2
WAN22_MODEL_DIR="${MODEL_BASE_PATH}/Wan2.2-TI2V-5B"
# Keep T5 + tokenizer consistent with Stage 1 (WAN2.2-TI2V-5B bundle).
WAN21_T5_MODEL="${WAN22_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
WAN21_TOKENIZER_DIR="${WAN22_MODEL_DIR}/google/umt5-xxl"

# Prefer nested DiT folder when available.
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
echo "Stage 2: LoRA Fine-tuning with Pre-trained Heads"
echo "=========================================================================="
echo "Loading checkpoints from Stage 1:"
echo "  Motion head: $MOTION_HEAD_CHECKPOINT"
echo "  Depth head:  $DEPTH_HEAD_CHECKPOINT"
echo "=========================================================================="

# LoRA configuration
LORA_RANK=32
LORA_TARGET_MODULES="q,k,v,o,ffn.0,ffn.2"

# Training hyperparameters (lower LR for fine-tuning)
LEARNING_RATE=1e-5
NUM_EPOCHS=5
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
SAVE_STEPS=500

# Loss weights (can be adjusted for Stage 2)
MOTION_LOSS_WEIGHT=0.5
DEPTH_LOSS_WEIGHT=0.5

# Wandb configuration (optional)
USE_WANDB=true
WANDB_PROJECT="wan22-ti2v-stage2-lora"

# Optional: Use spatio-temporal depth head for better temporal consistency
USE_SPATIOTEMPORAL_DEPTH=false
SPATIOTEMPORAL_DEPTH_TYPE="simple"  # or "full"

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
  $([ "$USE_WANDB" = true ] && echo "--use_wandb --wandb_project $WANDB_PROJECT" || echo "") \
  $([ "$USE_SPATIOTEMPORAL_DEPTH" = true ] && echo "--use_spatiotemporal_depth --spatiotemporal_depth_type $SPATIOTEMPORAL_DEPTH_TYPE" || echo "")

echo ""
echo "=========================================================================="
echo "Stage 2 Training Complete!"
echo "=========================================================================="
echo "Final model has been saved to:"
echo "  $OUTPUT_PATH/final/motion_head.pth"
echo "  $OUTPUT_PATH/final/depth_head.pth"
echo "  $OUTPUT_PATH/final/lora_weights.pth"
echo ""
echo "You can now use these checkpoints for inference with text+image input!"
echo "=========================================================================="
