#!/bin/bash

# Batch test script for WAN2.2-TI2V Stage 2 finetuned checkpoints

FINETUNE_DIR="${1:-/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/full_data}"
OUTPUT_DIR="${2:-test_outputs_wan22_stage2_finetune}"
BASE_DIR="/nyx-storage1/hanliu/Sim_Physics/TestOutput"

DIT_CKPT=""
MOTION_HEAD=""
DEPTH_HEAD=""

if [ -f "${FINETUNE_DIR}/final/dit.pth" ]; then
    DIT_CKPT="${FINETUNE_DIR}/final/dit.pth"
elif [ -f "${FINETUNE_DIR}/dit.pth" ]; then
    DIT_CKPT="${FINETUNE_DIR}/dit.pth"
fi

if [ -f "${FINETUNE_DIR}/final/motion_head.pth" ]; then
    MOTION_HEAD="${FINETUNE_DIR}/final/motion_head.pth"
elif [ -f "${FINETUNE_DIR}/motion_head.pth" ]; then
    MOTION_HEAD="${FINETUNE_DIR}/motion_head.pth"
fi

if [ -f "${FINETUNE_DIR}/final/depth_head.pth" ]; then
    DEPTH_HEAD="${FINETUNE_DIR}/final/depth_head.pth"
elif [ -f "${FINETUNE_DIR}/depth_head.pth" ]; then
    DEPTH_HEAD="${FINETUNE_DIR}/depth_head.pth"
fi

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
echo "Batch Testing WAN2.2 Stage 2 Finetune"
echo "========================================="
echo "Finetune checkpoint: $DIT_CKPT"
echo "Motion head:         $MOTION_HEAD"
echo "Depth head:          $DEPTH_HEAD"
echo "Output directory:    $OUTPUT_DIR"
echo ""

if [ -z "$MOTION_HEAD" ] || [ ! -f "$MOTION_HEAD" ]; then
    echo "Error: Motion head checkpoint not found in: $FINETUNE_DIR"
    exit 1
fi

if [ -z "$DEPTH_HEAD" ] || [ ! -f "$DEPTH_HEAD" ]; then
    echo "Error: Depth head checkpoint not found in: $FINETUNE_DIR"
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
    OUTPUT_VIDEO_FINETUNE="$OUTPUT_DIR/${scenario}_finetune.mp4"
    OUTPUT_MOTION_BASE="$OUTPUT_DIR/${scenario}_base_motion.mp4"
    OUTPUT_DEPTH_BASE="$OUTPUT_DIR/${scenario}_base_depth.mp4"
    OUTPUT_MOTION_FINETUNE="$OUTPUT_DIR/${scenario}_finetune_motion.mp4"
    OUTPUT_DEPTH_FINETUNE="$OUTPUT_DIR/${scenario}_finetune_depth.mp4"

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

    if [ -n "$DIT_CKPT" ] && [ -f "$DIT_CKPT" ] && [ ! -f "$OUTPUT_VIDEO_FINETUNE" ]; then
        python eval_wan22_stage2_lora.py \
            --dit_checkpoint "$DIT_CKPT" \
            --motion_head_checkpoint "$MOTION_HEAD" \
            --depth_head_checkpoint "$DEPTH_HEAD" \
            --input_image "$INPUT_IMAGE" \
            --prompt "$prompt" \
            --output "$OUTPUT_VIDEO_FINETUNE" \
            --output_motion_video "$OUTPUT_MOTION_FINETUNE" \
            --output_depth_video "$OUTPUT_DEPTH_FINETUNE" \
            --height 480 \
            --width 480 \
            --num_frames 49 \
            --seed 42 \
            --fps 15

        if [ $? -eq 0 ]; then
            echo "  Finetune success: $OUTPUT_VIDEO_FINETUNE"
        else
            echo "  Finetune failed to generate video"
        fi
    elif [ -z "$DIT_CKPT" ] || [ ! -f "$DIT_CKPT" ]; then
        echo "  Finetune checkpoint missing, skipping finetune run..."
    else
        echo "  Finetune output already exists, skipping..."
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
