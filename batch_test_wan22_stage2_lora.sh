#!/bin/bash

# Batch test script for WAN2.2-TI2V Stage 2 LoRA checkpoints

LORA_DIR="${1:-/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage2/final}"
OUTPUT_DIR="${2:-test_outputs_wan22_stage2}"
BASE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"

LORA_CKPT="${LORA_DIR}/lora_weights.pth"
MOTION_HEAD="${LORA_DIR}/motion_head.pth"
DEPTH_HEAD="${LORA_DIR}/depth_head.pth"

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

run_batch_tests() {
echo "========================================="
echo "Batch Testing WAN2.2 Stage 2 LoRA"
echo "========================================="
echo "LoRA checkpoint: $LORA_CKPT"
echo "Motion head:     $MOTION_HEAD"
echo "Depth head:      $DEPTH_HEAD"
echo "Output directory: $OUTPUT_DIR"
echo ""

if [ ! -f "$LORA_CKPT" ]; then
    echo "Error: LoRA checkpoint not found: $LORA_CKPT"
    echo "Usage: $0 <stage2_final_dir> [output_dir]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOTAL=${#SCENARIOS[@]}
COUNT=0

echo "Testing $TOTAL scenarios..."
echo ""

for scenario in "${!SCENARIOS[@]}"; do
    COUNT=$((COUNT + 1))
    prompt="${SCENARIOS[$scenario]}"

    INPUT_IMAGE="$BASE_DIR/$scenario/env_0/0/0/rgb/rgb_0001.png"
    OUTPUT_VIDEO_BASE="$OUTPUT_DIR/${scenario}_base.mp4"
    OUTPUT_VIDEO_LORA="$OUTPUT_DIR/${scenario}_lora.mp4"
    OUTPUT_MOTION_BASE="$OUTPUT_DIR/${scenario}_base_motion.mp4"
    OUTPUT_DEPTH_BASE="$OUTPUT_DIR/${scenario}_base_depth.mp4"
    OUTPUT_MOTION_LORA="$OUTPUT_DIR/${scenario}_lora_motion.mp4"
    OUTPUT_DEPTH_LORA="$OUTPUT_DIR/${scenario}_lora_depth.mp4"

    echo "[$COUNT/$TOTAL] Testing: $scenario"
    echo "  Prompt: $prompt"

    if [ ! -f "$INPUT_IMAGE" ]; then
        echo "  Warning: Input image not found, skipping..."
        echo ""
        continue
    fi

    if [ ! -f "$OUTPUT_VIDEO_BASE" ]; then
        python eval_wan22_stage2_lora.py \
            --motion_head_checkpoint "$MOTION_HEAD" \
            --depth_head_checkpoint "$DEPTH_HEAD" \
            --input_image "$INPUT_IMAGE" \
            --prompt "$prompt" \
            --output "$OUTPUT_VIDEO_BASE" \
            --output_motion_video "$OUTPUT_MOTION_BASE" \
            --output_depth_video "$OUTPUT_DEPTH_BASE" \
            --height 480 \
            --width 480 \
            --num_frames 49 \
            --seed 42 \
            --fps 15

        if [ $? -eq 0 ]; then
            echo "  Base success: $OUTPUT_VIDEO_BASE"
        else
            echo "  Base failed to generate video"
        fi
    else
        echo "  Base output already exists, skipping..."
    fi

    if [ -f "$LORA_CKPT" ] && [ ! -f "$OUTPUT_VIDEO_LORA" ]; then
        python test_wan22_stage2_lora.py \
            --lora_checkpoint "$LORA_CKPT" \
            --motion_head_checkpoint "$MOTION_HEAD" \
            --depth_head_checkpoint "$DEPTH_HEAD" \
            --input_image "$INPUT_IMAGE" \
            --prompt "$prompt" \
            --output "$OUTPUT_VIDEO_LORA" \
            --output_motion_video "$OUTPUT_MOTION_LORA" \
            --output_depth_video "$OUTPUT_DEPTH_LORA" \
            --height 480 \
            --width 480 \
            --num_frames 49 \
            --seed 42 \
            --fps 15

        if [ $? -eq 0 ]; then
            echo "  LoRA success: $OUTPUT_VIDEO_LORA"
        else
            echo "  LoRA failed to generate video"
        fi
    elif [ ! -f "$LORA_CKPT" ]; then
        echo "  LoRA checkpoint missing, skipping LoRA run..."
    else
        echo "  LoRA output already exists, skipping..."
    fi
    echo ""
done

echo "========================================="
echo "Batch testing complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    run_batch_tests
fi
