#!/bin/bash

# Batch combine RGB/motion/depth videos into 1x3 layout videos.

OUTPUT_DIR="${1:-test_outputs_wan22_stage2}"
FPS="${2:-15}"
BATCH_SCRIPT="${3:-batch_test_wan22_stage2_lora.sh}"
OUTPUT_DIR_ORIG="$OUTPUT_DIR"
FPS_ORIG="$FPS"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "Error: Scenario script not found: $BATCH_SCRIPT"
    exit 1
fi

COMBINED_DIR="${OUTPUT_DIR}/combined"
mkdir -p "$COMBINED_DIR"

echo "========================================="
echo "Combining WAN2.2 outputs"
echo "Output dir: $OUTPUT_DIR"
echo "Combined dir: $COMBINED_DIR"
echo "FPS: $FPS"
echo "========================================="

source "$BATCH_SCRIPT"
OUTPUT_DIR="$OUTPUT_DIR_ORIG"
FPS="$FPS_ORIG"

for scenario in "${!SCENARIOS[@]}"; do
    for suffix in base lora; do
        rgb_video="${OUTPUT_DIR}/${scenario}_${suffix}.mp4"
        motion_video="${OUTPUT_DIR}/${scenario}_${suffix}_motion.mp4"
        depth_video="${OUTPUT_DIR}/${scenario}_${suffix}_depth.mp4"

        if [ ! -f "$rgb_video" ] || [ ! -f "$motion_video" ] || [ ! -f "$depth_video" ]; then
            echo "Skipping ${scenario}_${suffix} (missing rgb/motion/depth)"
            continue
        fi

        output_name="${scenario}_${suffix}_combined.mp4"
        output_path="${COMBINED_DIR}/${output_name}"
        if [ -f "$output_path" ]; then
            echo "Already exists: $output_path"
            continue
        fi

        echo "Combining: ${scenario}_${suffix}"
        python combine_wan22_outputs.py \
            --rgb_video "$rgb_video" \
            --motion_video "$motion_video" \
            --depth_video "$depth_video" \
            --output_dir "$COMBINED_DIR" \
            --output_name "$output_name" \
            --fps "$FPS"
    done
done

echo "Done."
