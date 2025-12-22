#!/bin/bash

###############################################################################
# Quick Start: WAN2.2-5B Two-Stage Training
#
# This script runs the complete two-stage training pipeline:
#   1. Stage 1: Train motion and depth heads (backbone frozen)
#   2. Stage 2: Fine-tune backbone with LoRA + trained heads
#
# Usage:
#   bash run_wan22_training.sh [stage1|stage2|all]
#
# Examples:
#   bash run_wan22_training.sh all       # Run both stages sequentially
#   bash run_wan22_training.sh stage1    # Run only stage 1
#   bash run_wan22_training.sh stage2    # Run only stage 2
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default: run both stages
MODE=${1:-all}

print_banner() {
    echo -e "${GREEN}"
    echo "=========================================================================="
    echo "$1"
    echo "=========================================================================="
    echo -e "${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

run_stage1() {
    print_banner "Starting Stage 1: Training Motion and Depth Heads"

    # Make script executable
    chmod +x train_wan22_stage1_heads.sh

    # Check if dataset exists
    DATASET_PATH=$(grep "DATASET_BASE_PATH=" train_wan22_stage1_heads.sh | cut -d'"' -f2)
    if [ ! -d "$DATASET_PATH" ]; then
        print_error "Dataset not found at: $DATASET_PATH"
        print_warning "Please update DATASET_BASE_PATH in train_wan22_stage1_heads.sh"
        exit 1
    fi

    # Run training
    ./train_wan22_stage1_heads.sh

    # Verify checkpoints were created
    if [ -f "./models/wan22_ti2v_stage1/final/motion_head.pth" ] && \
       [ -f "./models/wan22_ti2v_stage1/final/depth_head.pth" ]; then
        print_banner "Stage 1 Complete! ✓"
        echo "Checkpoints saved:"
        echo "  - ./models/wan22_ti2v_stage1/final/motion_head.pth"
        echo "  - ./models/wan22_ti2v_stage1/final/depth_head.pth"
    else
        print_error "Stage 1 checkpoints not found. Training may have failed."
        exit 1
    fi
}

run_stage2() {
    print_banner "Starting Stage 2: LoRA Fine-tuning with Trained Heads"

    # Make script executable
    chmod +x train_wan22_stage2_lora.sh

    # Verify Stage 1 checkpoints exist
    if [ ! -f "./models/wan22_ti2v_stage1/final/motion_head.pth" ]; then
        print_error "Motion head checkpoint not found!"
        print_warning "Please run Stage 1 first: bash run_wan22_training.sh stage1"
        exit 1
    fi

    if [ ! -f "./models/wan22_ti2v_stage1/final/depth_head.pth" ]; then
        print_error "Depth head checkpoint not found!"
        print_warning "Please run Stage 1 first: bash run_wan22_training.sh stage1"
        exit 1
    fi

    # Run training
    ./train_wan22_stage2_lora.sh

    # Verify checkpoints were created
    if [ -f "./models/wan22_ti2v_stage2/final/lora_weights.pth" ]; then
        print_banner "Stage 2 Complete! ✓"
        echo "Final model checkpoints saved:"
        echo "  - ./models/wan22_ti2v_stage2/final/motion_head.pth"
        echo "  - ./models/wan22_ti2v_stage2/final/depth_head.pth"
        echo "  - ./models/wan22_ti2v_stage2/final/lora_weights.pth"
    else
        print_error "Stage 2 checkpoints not found. Training may have failed."
        exit 1
    fi
}

main() {
    case $MODE in
        stage1)
            run_stage1
            ;;
        stage2)
            run_stage2
            ;;
        all)
            run_stage1
            echo ""
            echo "Waiting 5 seconds before starting Stage 2..."
            sleep 5
            run_stage2

            print_banner "Complete Two-Stage Training Finished! 🎉"
            echo "You can now use the trained model for inference."
            echo ""
            echo "See WAN22_TI2V_TRAINING_GUIDE.md for inference instructions."
            ;;
        *)
            print_error "Invalid mode: $MODE"
            echo "Usage: bash run_wan22_training.sh [stage1|stage2|all]"
            exit 1
            ;;
    esac
}

# Show configuration
print_banner "WAN2.2-5B Two-Stage Training Pipeline"
echo "Mode: $MODE"
echo ""
echo "Before starting, please ensure:"
echo "  1. Dataset is prepared with videos, motion flows, and depth maps"
echo "  2. Paths are configured in train_wan22_stage1_heads.sh and train_wan22_stage2_lora.sh"
echo "  3. You have enough GPU memory (recommended: 4x A100 40GB or similar)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

main
