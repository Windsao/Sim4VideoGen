#!/bin/bash

# Fine-tuning script for WAN model with LoRA + Motion + Depth + RGB losses
#
# This script loads checkpoints from train_wan_with_motion.py (motion and depth heads)
# and continues fine-tuning with LoRA while training all three components:
# 1. RGB/Noise prediction loss (standard diffusion via LoRA)
# 2. Motion vector prediction loss (motion head)
# 3. Depth prediction loss (depth head)
#
# Usage:
#   ./train_wan_finetune_all.sh [checkpoint_dir] [options]
#
# Arguments:
#   checkpoint_dir   - Directory containing motion_head.pt and depth_head.pt from stage 1
#   --from_scratch   - Start from scratch without loading pre-trained heads
#   --freeze_heads   - Freeze motion and depth heads (only train LoRA)

# ============================================
# Configuration
# ============================================

# Default paths
SOURCE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
OUTPUT_METADATA_DIR="data/sim_physics_motion_dataset"
MODEL_BASE_PATH="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI"

# Checkpoint directory from stage 1 (train_wan_with_motion.py)
# This should contain motion_head.pt and depth_head.pt
STAGE1_CHECKPOINT_DIR=""
FROM_SCRATCH=false
FREEZE_HEADS=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --from_scratch)
            FROM_SCRATCH=true
            ;;
        --freeze_heads)
            FREEZE_HEADS=true
            ;;
        *)
            # Assume it's the checkpoint directory
            if [ -d "$arg" ]; then
                STAGE1_CHECKPOINT_DIR="$arg"
            else
                echo "Warning: '$arg' is not a valid directory, ignoring..."
            fi
            ;;
    esac
done

# Output path for fine-tuned model
OUTPUT_MODEL_PATH="${MODEL_BASE_PATH}/sim_physics_Wan2.1_finetune_all"

# Training hyperparameters
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
DATASET_REPEAT=100
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LORA_RANK=32
GRADIENT_ACCUMULATION_STEPS=4

# Loss weights - balance all three losses
NOISE_LOSS_WEIGHT=1.0
MOTION_LOSS_WEIGHT=0.1
DEPTH_LOSS_WEIGHT=0.1

# Motion parameters
MOTION_CHANNELS=4
MOTION_LOSS_TYPE="mse"
MOTION_SCALE=0.01

# Depth parameters
DEPTH_LOSS_TYPE="mse"
DEPTH_SCALE=1.0

# Warp loss (temporal consistency)
USE_WARP_LOSS=true
WARP_LOSS_WEIGHT=0.1
WARP_LOSS_TYPE="mse"
if [ "${USE_WARP_LOSS}" = true ]; then
    OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH}_warp"
fi

# Spatio-temporal depth head
USE_SPATIOTEMPORAL_DEPTH=true
SPATIOTEMPORAL_DEPTH_TYPE="full"  # "simple" or "full"
NUM_TEMPORAL_HEADS=8
TEMPORAL_HEAD_DIM=64
NUM_TEMPORAL_BLOCKS=2
TEMPORAL_POS_EMBED_TYPE="rope"
if [ "${USE_SPATIOTEMPORAL_DEPTH}" = true ]; then
    if [ "${SPATIOTEMPORAL_DEPTH_TYPE}" = "simple" ]; then
        OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH}_head_simple"
    else
        OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH}_head_full"
    fi
fi

# Timestep sampling strategy
# Options: uniform, high_noise_bias, logit_normal, logit_normal_high,
#          cubic_high, linear_high, beta_high, truncated_high
# "high_noise_bias" samples more from high timesteps (noisy images) which helps
# improve the model's ability to handle early denoising steps
TIMESTEP_SAMPLING="high_noise_bias"  # Recommended for better denoising
TIMESTEP_BIAS_STRENGTH=2.0  # Higher = more high noise samples (only for high_noise_bias)
if [ "${TIMESTEP_SAMPLING}" != "uniform" ]; then
    OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH}_${TIMESTEP_SAMPLING}"
fi

# Checkpoint loading options
MOTION_HEAD_CHECKPOINT=""
DEPTH_HEAD_CHECKPOINT=""
if [ -n "${STAGE1_CHECKPOINT_DIR}" ] && [ "${FROM_SCRATCH}" = false ]; then
    # Look for checkpoint files (prefer .safetensors, fallback to .pt)
    # Try .safetensors first
    if [ -f "${STAGE1_CHECKPOINT_DIR}/motion_head.safetensors" ]; then
        MOTION_HEAD_CHECKPOINT="${STAGE1_CHECKPOINT_DIR}/motion_head.safetensors"
    elif ls ${STAGE1_CHECKPOINT_DIR}/motion_head_step-*.safetensors 1>/dev/null 2>&1; then
        # Find the latest step checkpoint
        MOTION_HEAD_CHECKPOINT=$(ls -t ${STAGE1_CHECKPOINT_DIR}/motion_head_step-*.safetensors 2>/dev/null | head -1)
    # Fallback to .pt format
    elif [ -f "${STAGE1_CHECKPOINT_DIR}/motion_head.pt" ]; then
        MOTION_HEAD_CHECKPOINT="${STAGE1_CHECKPOINT_DIR}/motion_head.pt"
    elif ls ${STAGE1_CHECKPOINT_DIR}/motion_head_step-*.pt 1>/dev/null 2>&1; then
        MOTION_HEAD_CHECKPOINT=$(ls -t ${STAGE1_CHECKPOINT_DIR}/motion_head_step-*.pt 2>/dev/null | head -1)
    fi

    # Try .safetensors first for depth head
    if [ -f "${STAGE1_CHECKPOINT_DIR}/depth_head.safetensors" ]; then
        DEPTH_HEAD_CHECKPOINT="${STAGE1_CHECKPOINT_DIR}/depth_head.safetensors"
    elif ls ${STAGE1_CHECKPOINT_DIR}/depth_head_step-*.safetensors 1>/dev/null 2>&1; then
        DEPTH_HEAD_CHECKPOINT=$(ls -t ${STAGE1_CHECKPOINT_DIR}/depth_head_step-*.safetensors 2>/dev/null | head -1)
    # Fallback to .pt format
    elif [ -f "${STAGE1_CHECKPOINT_DIR}/depth_head.pt" ]; then
        DEPTH_HEAD_CHECKPOINT="${STAGE1_CHECKPOINT_DIR}/depth_head.pt"
    elif ls ${STAGE1_CHECKPOINT_DIR}/depth_head_step-*.pt 1>/dev/null 2>&1; then
        DEPTH_HEAD_CHECKPOINT=$(ls -t ${STAGE1_CHECKPOINT_DIR}/depth_head_step-*.pt 2>/dev/null | head -1)
    fi
fi

# Model paths
MODEL_PATHS="[\"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors\", \"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth\", \"${MODEL_BASE_PATH}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth\"]"

# ============================================
# Print Configuration
# ============================================

echo "========================================="
echo "WAN Fine-tune All (LoRA + Motion + Depth + RGB)"
echo "========================================="
echo ""
echo "Source directory: ${SOURCE_DIR}"
echo "Output model path: ${OUTPUT_MODEL_PATH}"
echo ""
echo "Loss weights:"
echo "  - Noise/RGB loss: ${NOISE_LOSS_WEIGHT}"
echo "  - Motion loss: ${MOTION_LOSS_WEIGHT}"
echo "  - Depth loss: ${DEPTH_LOSS_WEIGHT}"
echo ""
echo "Warp loss (temporal consistency):"
echo "  - Enabled: ${USE_WARP_LOSS}"
if [ "${USE_WARP_LOSS}" = true ]; then
    echo "  - Weight: ${WARP_LOSS_WEIGHT}"
    echo "  - Type: ${WARP_LOSS_TYPE}"
fi
echo ""
echo "Spatio-temporal depth head:"
echo "  - Enabled: ${USE_SPATIOTEMPORAL_DEPTH}"
if [ "${USE_SPATIOTEMPORAL_DEPTH}" = true ]; then
    echo "  - Type: ${SPATIOTEMPORAL_DEPTH_TYPE}"
    echo "  - Temporal heads: ${NUM_TEMPORAL_HEADS}"
    echo "  - Temporal blocks: ${NUM_TEMPORAL_BLOCKS}"
    echo "  - Position embedding: ${TEMPORAL_POS_EMBED_TYPE}"
fi
echo ""
echo "Checkpoint loading:"
if [ "${FROM_SCRATCH}" = true ]; then
    echo "  - Mode: From scratch (random initialization)"
else
    echo "  - Stage 1 checkpoint dir: ${STAGE1_CHECKPOINT_DIR:-'Not specified'}"
    echo "  - Motion head: ${MOTION_HEAD_CHECKPOINT:-'Not found'}"
    echo "  - Depth head: ${DEPTH_HEAD_CHECKPOINT:-'Not found'}"
fi
echo ""
echo "Timestep sampling (for high-noise training):"
echo "  - Strategy: ${TIMESTEP_SAMPLING}"
if [ "${TIMESTEP_SAMPLING}" = "high_noise_bias" ]; then
    echo "  - Bias strength: ${TIMESTEP_BIAS_STRENGTH}"
fi
echo ""
echo "Training options:"
echo "  - Freeze heads: ${FREEZE_HEADS}"
echo "  - LoRA rank: ${LORA_RANK}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Epochs: ${NUM_EPOCHS}"
echo ""

# ============================================
# Step 1: Prepare dataset metadata
# ============================================

echo "Step 1/2: Preparing dataset metadata..."
echo "Scanning all test scenarios in: ${SOURCE_DIR}"

mkdir -p "${OUTPUT_METADATA_DIR}"

python -c "
import os
import csv
import glob

source_dir = '${SOURCE_DIR}'
output_dir = '${OUTPUT_METADATA_DIR}'

entries = []

for root, dirs, files in os.walk(source_dir):
    rgb_dir = os.path.join(root, 'rgb')
    motion_dir = os.path.join(root, 'motion_vectors')
    depth_dir = os.path.join(root, 'distance_to_camera')

    if os.path.isdir(rgb_dir) and os.path.isdir(motion_dir):
        rgb_files = glob.glob(os.path.join(rgb_dir, '*.png'))
        motion_files = glob.glob(os.path.join(motion_dir, '*.npy'))

        if len(rgb_files) >= 10 and len(motion_files) >= 10:
            rel_path = os.path.relpath(root, source_dir)
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
# Step 2: Run fine-tuning
# ============================================

echo "Step 2/2: Starting fine-tuning with LoRA + Motion + Depth + RGB..."
echo ""

# Build checkpoint arguments
CHECKPOINT_ARGS=""
if [ -n "${MOTION_HEAD_CHECKPOINT}" ]; then
    CHECKPOINT_ARGS="${CHECKPOINT_ARGS} --motion_head_checkpoint ${MOTION_HEAD_CHECKPOINT}"
fi
if [ -n "${DEPTH_HEAD_CHECKPOINT}" ]; then
    CHECKPOINT_ARGS="${CHECKPOINT_ARGS} --depth_head_checkpoint ${DEPTH_HEAD_CHECKPOINT}"
fi

# Build freeze arguments
FREEZE_ARGS=""
if [ "${FREEZE_HEADS}" = true ]; then
    FREEZE_ARGS="--freeze_motion_head --freeze_depth_head"
fi

# Build spatio-temporal arguments
SPATIOTEMPORAL_ARGS=""
if [ "${USE_SPATIOTEMPORAL_DEPTH}" = true ]; then
    SPATIOTEMPORAL_ARGS="--use_spatiotemporal_depth --spatiotemporal_depth_type ${SPATIOTEMPORAL_DEPTH_TYPE} --num_temporal_heads ${NUM_TEMPORAL_HEADS} --temporal_head_dim ${TEMPORAL_HEAD_DIM} --num_temporal_blocks ${NUM_TEMPORAL_BLOCKS} --temporal_pos_embed_type ${TEMPORAL_POS_EMBED_TYPE}"
fi

# Build warp loss arguments
WARP_ARGS=""
if [ "${USE_WARP_LOSS}" = true ]; then
    WARP_ARGS="--use_warp_loss --warp_loss_weight ${WARP_LOSS_WEIGHT} --warp_loss_type ${WARP_LOSS_TYPE}"
fi

# Generate wandb run name
WANDB_RUN_NAME="finetune_all"
if [ "${FROM_SCRATCH}" = true ]; then
    WANDB_RUN_NAME="${WANDB_RUN_NAME}_scratch"
else
    WANDB_RUN_NAME="${WANDB_RUN_NAME}_pretrained"
fi
if [ "${FREEZE_HEADS}" = true ]; then
    WANDB_RUN_NAME="${WANDB_RUN_NAME}_freeze"
fi
if [ "${TIMESTEP_SAMPLING}" != "uniform" ]; then
    WANDB_RUN_NAME="${WANDB_RUN_NAME}_${TIMESTEP_SAMPLING}"
fi

accelerate launch train_wan_finetune_all.py \
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
    --noise_loss_weight ${NOISE_LOSS_WEIGHT} \
    --motion_channels ${MOTION_CHANNELS} \
    --motion_loss_weight ${MOTION_LOSS_WEIGHT} \
    --motion_loss_type ${MOTION_LOSS_TYPE} \
    --motion_scale ${MOTION_SCALE} \
    --depth_loss_weight ${DEPTH_LOSS_WEIGHT} \
    --depth_loss_type ${DEPTH_LOSS_TYPE} \
    --depth_scale ${DEPTH_SCALE} \
    ${CHECKPOINT_ARGS} \
    ${FREEZE_ARGS} \
    ${SPATIOTEMPORAL_ARGS} \
    ${WARP_ARGS} \
    --save_heads_every_n_steps 100 \
    --timestep_sampling ${TIMESTEP_SAMPLING} \
    --timestep_bias_strength ${TIMESTEP_BIAS_STRENGTH} \
    --wandb_project "Sim4Videos" \
    --wandb_run_name ${WANDB_RUN_NAME} \


if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Fine-tuning complete!"
echo "========================================="
echo "LoRA model saved to: ${OUTPUT_MODEL_PATH}"
echo "Motion head saved to: ${OUTPUT_MODEL_PATH}/motion_head*.safetensors"
echo "Depth head saved to: ${OUTPUT_MODEL_PATH}/depth_head*.safetensors"
echo ""
echo "To run inference:"
echo "  python inference_wan_with_motion.py \\"
echo "      --lora_checkpoint ${OUTPUT_MODEL_PATH}/step-XXXX.safetensors \\"
echo "      --motion_head_checkpoint ${OUTPUT_MODEL_PATH}/motion_head_step-XXXX.safetensors \\"
echo "      --depth_head_checkpoint ${OUTPUT_MODEL_PATH}/depth_head_step-XXXX.safetensors \\"
echo "      --prompt 'Your prompt here' \\"
echo "      --output output.mp4"
echo ""
