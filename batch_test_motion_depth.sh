#!/bin/bash

# Batch test script for WAN model with motion and depth heads
# Tests the model on multiple physics scenarios and generates motion/depth predictions
#
# Usage:
#   ./batch_test_motion_depth.sh [lora_ckpt] [motion_head_ckpt] [depth_head_ckpt] [output_dir]
#
# Examples:
#   # Test with motion head only
#   ./batch_test_motion_depth.sh "" /path/to/motion_head/step-2000.safetensors
#
#   # Test with both LoRA and motion head
#   ./batch_test_motion_depth.sh /path/to/lora.safetensors /path/to/motion_head.safetensors
#
#   # Test with all heads
#   ./batch_test_motion_depth.sh /path/to/lora.safetensors /path/to/motion.safetensors /path/to/depth.safetensors

# Configuration
LORA_CKPT="${1:-}"
MOTION_HEAD_CKPT="${2:-/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/sim_physics_Wan2.1_motion_lora/step-2000.safetensors}"
DEPTH_HEAD_CKPT="${3:-}"
BASE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"
OUTPUT_DIR="${4:-test_outputs_motion_lora}"

echo "========================================="
echo "Batch Testing WAN Model with Motion/Depth"
echo "========================================="
echo "LoRA checkpoint: ${LORA_CKPT:-none}"
echo "Motion head checkpoint: ${MOTION_HEAD_CKPT:-none}"
echo "Depth head checkpoint: ${DEPTH_HEAD_CKPT:-none}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Validate checkpoints
if [ -n "$LORA_CKPT" ] && [ ! -f "$LORA_CKPT" ]; then
    echo "Warning: LoRA checkpoint not found: $LORA_CKPT"
fi

if [ -n "$MOTION_HEAD_CKPT" ] && [ ! -f "$MOTION_HEAD_CKPT" ]; then
    echo "Warning: Motion head checkpoint not found: $MOTION_HEAD_CKPT"
fi

if [ -n "$DEPTH_HEAD_CKPT" ] && [ ! -f "$DEPTH_HEAD_CKPT" ]; then
    echo "Warning: Depth head checkpoint not found: $DEPTH_HEAD_CKPT"
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
    OUTPUT_MOTION="$OUTPUT_DIR/${scenario}_motion.npy"
    OUTPUT_DEPTH="$OUTPUT_DIR/${scenario}_depth.npy"
    OUTPUT_MOTION_VIS="$OUTPUT_DIR/${scenario}_motion_vis.mp4"
    OUTPUT_DEPTH_VIS="$OUTPUT_DIR/${scenario}_depth_vis.mp4"

    echo "[$COUNT/$TOTAL] Testing: $scenario"
    echo "  Prompt: $prompt"

    if [ ! -f "$INPUT_IMAGE" ]; then
        echo "  Warning: Input image not found, skipping..."
        echo ""
        continue
    fi

    if [ -f "$OUTPUT_VIDEO" ]; then
        echo "  Output already exists, skipping..."
        echo ""
        continue
    fi

    # Build command with optional arguments
    CMD="python inference_wan_motion_depth.py"
    CMD="$CMD --input_image \"$INPUT_IMAGE\""
    CMD="$CMD --prompt \"$prompt\""
    CMD="$CMD --output \"$OUTPUT_VIDEO\""
    CMD="$CMD --height 480"
    CMD="$CMD --width 832"
    CMD="$CMD --num_frames 81"
    CMD="$CMD --seed 42"
    CMD="$CMD --fps 15"

    # Add LoRA if provided
    if [ -n "$LORA_CKPT" ] && [ -f "$LORA_CKPT" ]; then
        CMD="$CMD --lora_checkpoint \"$LORA_CKPT\""
    fi

    # Add motion head if provided
    if [ -n "$MOTION_HEAD_CKPT" ] && [ -f "$MOTION_HEAD_CKPT" ]; then
        CMD="$CMD --motion_head_checkpoint \"$MOTION_HEAD_CKPT\""
        CMD="$CMD --output_motion \"$OUTPUT_MOTION\""
        CMD="$CMD --output_motion_vis \"$OUTPUT_MOTION_VIS\""
    fi

    # Add depth head if provided
    if [ -n "$DEPTH_HEAD_CKPT" ] && [ -f "$DEPTH_HEAD_CKPT" ]; then
        CMD="$CMD --depth_head_checkpoint \"$DEPTH_HEAD_CKPT\""
        CMD="$CMD --output_depth \"$OUTPUT_DEPTH\""
        CMD="$CMD --output_depth_vis \"$OUTPUT_DEPTH_VIS\""
    fi

    # Run inference
    eval $CMD

    if [ $? -eq 0 ]; then
        echo "  Success: $OUTPUT_VIDEO"
        [ -f "$OUTPUT_MOTION" ] && echo "  Motion vectors: $OUTPUT_MOTION"
        [ -f "$OUTPUT_DEPTH" ] && echo "  Depth maps: $OUTPUT_DEPTH"
    else
        echo "  Failed to generate video"
    fi
    echo ""
done

echo "========================================="
echo "Batch testing complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs echo "  Videos:"
ls -lh "$OUTPUT_DIR"/*_motion.npy 2>/dev/null | wc -l | xargs echo "  Motion vectors:"
ls -lh "$OUTPUT_DIR"/*_depth.npy 2>/dev/null | wc -l | xargs echo "  Depth maps:"
echo ""
echo "View results with:"
echo "  ls -lh $OUTPUT_DIR/"
