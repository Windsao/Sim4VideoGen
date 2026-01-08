#!/bin/bash

# Batch test script for WAN2.2-TI2V Stage 2 finetuned checkpoints
# using full-resolution motion/depth heads.

FINETUNE_DIR="${1:-/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/full_data}"
OUTPUT_DIR="${2:-test_outputs_wan22_stage2_finetune_fullres}"

# Spatio-temporal depth head (VDA-style) configuration
USE_SPATIOTEMPORAL_DEPTH=true
SPATIOTEMPORAL_DEPTH_TYPE="full" # "simple" or "full"
NUM_TEMPORAL_HEADS=8
TEMPORAL_HEAD_DIM=64
NUM_TEMPORAL_BLOCKS=2
TEMPORAL_POS_EMBED_TYPE="rope"   # "rope" or "ape"

# Spatio-temporal motion head (VDA-style) configuration
USE_SPATIOTEMPORAL_MOTION=true
SPATIOTEMPORAL_MOTION_TYPE="full" # "simple" or "full"
FULL_RES_UPSAMPLE_MODE="trilinear"

DEFAULT_HEADS_DIR="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage1_fullres_large/final"
if [ "$USE_SPATIOTEMPORAL_DEPTH" = true ]; then
    if [ "$SPATIOTEMPORAL_DEPTH_TYPE" = "full" ]; then
        DEFAULT_HEADS_DIR="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage1_spatio_depth_full/final"
    else
        DEFAULT_HEADS_DIR="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage1_spatio_depth_simple/final"
    fi
fi

HEADS_DIR="${3:-$DEFAULT_HEADS_DIR}"
METADATA_CSV="/home/mzh1800/DiffSynth-Studio/data/magic_physics_dataset/metadata.csv"

DIT_CKPT=""
MOTION_HEAD=""
DEPTH_HEAD=""

HEIGHT=480
WIDTH=480
NUM_FRAMES=49

if [ -f "${FINETUNE_DIR}/final/dit.pth" ]; then
    DIT_CKPT="${FINETUNE_DIR}/final/dit.pth"
elif [ -f "${FINETUNE_DIR}/dit.pth" ]; then
    DIT_CKPT="${FINETUNE_DIR}/dit.pth"
fi

if [ -f "${HEADS_DIR}/final/motion_head.pth" ]; then
    MOTION_HEAD="${HEADS_DIR}/final/motion_head.pth"
elif [ -f "${HEADS_DIR}/motion_head.pth" ]; then
    MOTION_HEAD="${HEADS_DIR}/motion_head.pth"
fi

if [ -f "${HEADS_DIR}/final/depth_head.pth" ]; then
    DEPTH_HEAD="${HEADS_DIR}/final/depth_head.pth"
elif [ -f "${HEADS_DIR}/depth_head.pth" ]; then
    DEPTH_HEAD="${HEADS_DIR}/depth_head.pth"
fi

SCENARIOS=(
    "test_ball_and_block_collect"
    "test_ball_collide_collect"
    "test_ball_hits_duck_collect"
    "test_ball_hits_nothing_collect"
    "test_ball_in_basket_collect"
    "test_ball_ramp_collect"
    "test_ball_rolls_on_glass_collect"
    "test_ball_train_collect"
)

lookup_prompt_and_video() {
python - "$METADATA_CSV" "$1" <<'PY'
import csv
import sys

metadata_path = sys.argv[1]
scenario = sys.argv[2]

with open(metadata_path, newline="") as f:
    for row in csv.DictReader(f):
        video = row.get("video", "")
        if f"/{scenario}/" in video:
            print(row.get("prompt", ""))
            print(video)
            sys.exit(0)

print("")
print("")
PY
}

run_batch_tests() {
echo "========================================="
echo "Batch Testing WAN2.2 Stage 2 Finetune (Full-Res Heads)"
echo "========================================="
echo "Finetune checkpoint: $DIT_CKPT"
echo "Motion head:         $MOTION_HEAD"
echo "Depth head:          $DEPTH_HEAD"
echo "Output directory:    $OUTPUT_DIR"

echo ""

if [ -z "$MOTION_HEAD" ] || [ ! -f "$MOTION_HEAD" ]; then
    echo "Error: Full-res motion head checkpoint not found in: $HEADS_DIR"
    exit 1
fi

if [ -z "$DEPTH_HEAD" ] || [ ! -f "$DEPTH_HEAD" ]; then
    echo "Error: Full-res depth head checkpoint not found in: $HEADS_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

TOTAL=${#SCENARIOS[@]}
COUNT=0

echo "Testing $TOTAL scenarios..."
echo ""

for scenario in "${SCENARIOS[@]}"; do
    COUNT=$((COUNT + 1))
    mapfile -t meta < <(lookup_prompt_and_video "$scenario")
    prompt="${meta[0]}"
    video_path="${meta[1]}"

    if [ -z "$prompt" ] || [ -z "$video_path" ]; then
        echo "  Warning: Prompt not found in metadata for scenario: $scenario"
        echo ""
        continue
    fi

    INPUT_IMAGE=""
    if [ -d "$video_path/rgb_capture" ]; then
        INPUT_IMAGE=$(ls -1 "$video_path/rgb_capture"/step_*.png 2>/dev/null | head -n 1)
    fi
    OUTPUT_VIDEO_BASE="$OUTPUT_DIR/${scenario}_base.mp4"
    OUTPUT_VIDEO_FINETUNE="$OUTPUT_DIR/${scenario}_finetune.mp4"
    OUTPUT_MOTION_BASE="$OUTPUT_DIR/${scenario}_base_motion.mp4"
    OUTPUT_DEPTH_BASE="$OUTPUT_DIR/${scenario}_base_depth.mp4"
    OUTPUT_MOTION_FINETUNE="$OUTPUT_DIR/${scenario}_finetune_motion.mp4"
    OUTPUT_DEPTH_FINETUNE="$OUTPUT_DIR/${scenario}_finetune_depth.mp4"

    echo "[$COUNT/$TOTAL] Testing: $scenario"
    echo "  Prompt: $prompt"

    if [ -z "$INPUT_IMAGE" ] || [ ! -f "$INPUT_IMAGE" ]; then
        echo "  Warning: Input image not found, skipping..."
        echo "  Video path: $video_path"
        echo "  Expected: $video_path/rgb_capture/step_*.png"
        echo ""
        continue
    fi

    if [ ! -f "$OUTPUT_VIDEO_BASE" ]; then
        python eval_wan22_stage2_lora.py \
            --motion_head_checkpoint "$MOTION_HEAD" \
            --depth_head_checkpoint "$DEPTH_HEAD" \
            --full_res_heads \
            --full_res_upsample_mode "$FULL_RES_UPSAMPLE_MODE" \
            $([ "$USE_SPATIOTEMPORAL_DEPTH" = true ] && echo "--use_spatiotemporal_depth --spatiotemporal_depth_type $SPATIOTEMPORAL_DEPTH_TYPE --num_temporal_heads $NUM_TEMPORAL_HEADS --temporal_head_dim $TEMPORAL_HEAD_DIM --num_temporal_blocks $NUM_TEMPORAL_BLOCKS --temporal_pos_embed_type $TEMPORAL_POS_EMBED_TYPE" || echo "") \
            $([ "$USE_SPATIOTEMPORAL_MOTION" = true ] && echo "--use_spatiotemporal_motion --spatiotemporal_motion_type $SPATIOTEMPORAL_MOTION_TYPE" || echo "") \
            --input_image "$INPUT_IMAGE" \
            --prompt "$prompt" \
            --output "$OUTPUT_VIDEO_BASE" \
            --output_motion_video "$OUTPUT_MOTION_BASE" \
            --output_depth_video "$OUTPUT_DEPTH_BASE" \
            --height $HEIGHT \
            --width $WIDTH \
            --vis_width $WIDTH \
            --vis_height $HEIGHT \
            --num_frames $NUM_FRAMES \
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
            --full_res_heads \
            --full_res_upsample_mode "$FULL_RES_UPSAMPLE_MODE" \
            $([ "$USE_SPATIOTEMPORAL_DEPTH" = true ] && echo "--use_spatiotemporal_depth --spatiotemporal_depth_type $SPATIOTEMPORAL_DEPTH_TYPE --num_temporal_heads $NUM_TEMPORAL_HEADS --temporal_head_dim $TEMPORAL_HEAD_DIM --num_temporal_blocks $NUM_TEMPORAL_BLOCKS --temporal_pos_embed_type $TEMPORAL_POS_EMBED_TYPE" || echo "") \
            $([ "$USE_SPATIOTEMPORAL_MOTION" = true ] && echo "--use_spatiotemporal_motion --spatiotemporal_motion_type $SPATIOTEMPORAL_MOTION_TYPE" || echo "") \
            --input_image "$INPUT_IMAGE" \
            --prompt "$prompt" \
            --output "$OUTPUT_VIDEO_FINETUNE" \
            --output_motion_video "$OUTPUT_MOTION_FINETUNE" \
            --output_depth_video "$OUTPUT_DEPTH_FINETUNE" \
            --height $HEIGHT \
            --width $WIDTH \
            --vis_width $WIDTH \
            --vis_height $HEIGHT \
            --num_frames $NUM_FRAMES \
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
