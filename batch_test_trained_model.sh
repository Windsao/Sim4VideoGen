#!/bin/bash

# Batch test script for trained WAN model
# Tests the model on multiple physics scenarios

# Configuration
LORA_CKPT="${1:-/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/sim_physics_Wan2.2_lora/step-1500.safetensors}"
BASE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
OUTPUT_DIR="${2:-test_outputs_trained_model_1500}"

echo "========================================="
echo "Batch Testing Trained WAN Model"
echo "========================================="
echo "LoRA checkpoint: $LORA_CKPT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$LORA_CKPT" ]; then
    echo "Error: LoRA checkpoint not found: $LORA_CKPT"
    echo ""
    echo "Usage: $0 <lora_checkpoint> [output_dir]"
    echo "Example: $0 /path/to/step-2600.safetensors test_outputs"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test scenarios with their prompts
declare -A SCENARIOS=(
    ["test_ball_and_block_fall"]="A physics simulation of a ball and block falling"
    ["test_ball_collide"]="A physics simulation showing balls colliding"
    ["test_ball_hits_duck"]="A physics simulation of a ball hitting a duck"
    ["test_ball_ramp"]="A physics simulation of a ball rolling down a ramp"
    ["test_block_domino"]="A physics simulation of blocks falling like dominoes"
    ["test_duck_falls_in_box"]="A physics simulation of a duck falling into a box"
    ["test_ball_in_basket"]="A physics simulation of a ball going into a basket"
    ["test_ball_rolls_off"]="A physics simulation of a ball rolling off a surface"
)

TOTAL=${#SCENARIOS[@]}
COUNT=0

echo "Testing $TOTAL scenarios..."
echo ""

for scenario in "${!SCENARIOS[@]}"; do
    COUNT=$((COUNT + 1))
    prompt="${SCENARIOS[$scenario]}"

    INPUT_IMAGE="$BASE_DIR/$scenario/env_0/0/0/rgb/rgb_0001.png"
    OUTPUT_VIDEO="$OUTPUT_DIR/${scenario}.mp4"

    echo "[$COUNT/$TOTAL] Testing: $scenario"
    echo "  Prompt: $prompt"

    if [ ! -f "$INPUT_IMAGE" ]; then
        echo "  ⚠ Warning: Input image not found, skipping..."
        echo ""
        continue
    fi

    if [ -f "$OUTPUT_VIDEO" ]; then
        echo "  ℹ Output already exists, skipping..."
        echo ""
        continue
    fi

    python test_trained_model.py \
        --lora_checkpoint "$LORA_CKPT" \
        --input_image "$INPUT_IMAGE" \
        --prompt "$prompt" \
        --output "$OUTPUT_VIDEO" \
        --height 480 \
        --width 832 \
        --num_frames 81 \
        --seed 42 \
        --fps 15

    if [ $? -eq 0 ]; then
        echo "  ✓ Success: $OUTPUT_VIDEO"
    else
        echo "  ✗ Failed to generate video"
    fi
    echo ""
done

echo "========================================="
echo "Batch testing complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated videos:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs echo "Total videos:"
echo ""
echo "View results with:"
echo "  ls -lh $OUTPUT_DIR/"
