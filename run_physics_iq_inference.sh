#!/bin/bash

# Physics-IQ Benchmark Inference Script for Wan2.2-TI2V-5B
# This script runs the Wan2.2-TI2V-5B model on Physics-IQ benchmark test cases

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Physics-IQ Benchmark Wan2.2-TI2V-5B Pipeline${NC}"
echo -e "${BLUE}================================================${NC}"

# Configuration
OUTPUT_DIR="physics_iq_results"
MODEL_NAME="wan22_ti2v_5b"
MAX_SAMPLES=""  # Leave empty for all samples, or set a number like "10"
NUM_FRAMES=81   # 81 frames at 16fps = ~5 seconds (benchmark requirement)
FPS=16          # Benchmark evaluates first 5 seconds
HEIGHT=480
WIDTH=832
CFG_SCALE=7.0
INFERENCE_STEPS=50
SEED=42
DEVICE="cuda"
SHARD_INDEX=""
NUM_SHARDS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --cfg-scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --inference-steps)
            INFERENCE_STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resize-input)
            RESIZE_INPUT="--resize_input"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --shard-index)
            SHARD_INDEX="$2"
            shift 2
            ;;
        --num-shards)
            NUM_SHARDS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --max-samples N        Process only N samples (default: all)"
            echo "  --output-dir DIR       Output directory (default: physics_iq_results)"
            echo "  --model-name NAME      Model name for output folder (default: wan22)"
            echo "  --num-frames N         Number of frames to generate (default: 81)"
            echo "  --fps N                Output video FPS (default: 16)"
            echo "  --height N             Video height (default: 480)"
            echo "  --width N              Video width (default: 832)"
            echo "  --cfg-scale N          CFG scale (default: 7.0)"
            echo "  --inference-steps N    Inference steps (default: 50)"
            echo "  --seed N               Random seed (default: 42)"
            echo "  --resize-input         Resize input images to output dimensions"
            echo "  --device DEVICE        Device to use (default: cuda)"
            echo "  --shard-index N        Shard index for multi-GPU (0 to num-shards-1)"
            echo "  --num-shards N         Total number of shards for multi-GPU (default: 1)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run on first 10 samples"
            echo "  $0 --max-samples 10"
            echo ""
            echo "  # Run with custom output directory and model name"
            echo "  $0 --output-dir my_results --model-name wan22_custom"
            echo ""
            echo "  # Run with higher quality settings"
            echo "  $0 --inference-steps 100 --cfg-scale 8.0"
            echo ""
            echo "  # Multi-GPU: Run shard 0 of 4 on GPU 0"
            echo "  CUDA_VISIBLE_DEVICES=0 $0 --shard-index 0 --num-shards 4"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model Name: $MODEL_NAME"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  Max Samples: $MAX_SAMPLES"
else
    echo "  Max Samples: All samples"
fi
echo "  Video Resolution: ${WIDTH}x${HEIGHT}"
echo "  Frames: $NUM_FRAMES at ${FPS}fps (~$(echo "scale=1; $NUM_FRAMES / $FPS" | bc)s)"
echo "  CFG Scale: $CFG_SCALE"
echo "  Inference Steps: $INFERENCE_STEPS"
echo "  Seed: $SEED"
echo "  Device: $DEVICE"
if [ -n "$RESIZE_INPUT" ]; then
    echo "  Resize Input: Yes"
else
    echo "  Resize Input: No"
fi
if [ -n "$SHARD_INDEX" ]; then
    echo "  Shard: $SHARD_INDEX of $NUM_SHARDS"
fi
echo ""

# Check if Python script exists
SCRIPT_PATH="inference_physics_iq_wan22.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: $SCRIPT_PATH not found!${NC}"
    echo "Please ensure the inference script is in the current directory."
    exit 1
fi

# Check GPU availability
if [ "$DEVICE" == "cuda" ]; then
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Warning: NVIDIA GPU not detected. Falling back to CPU.${NC}"
        DEVICE="cpu"
    else
        echo -e "${GREEN}GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        echo ""
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build the command
CMD="python $SCRIPT_PATH"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --model_name $MODEL_NAME"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi
CMD="$CMD --height $HEIGHT"
CMD="$CMD --width $WIDTH"
CMD="$CMD --num_frames $NUM_FRAMES"
CMD="$CMD --fps $FPS"
CMD="$CMD --cfg_scale $CFG_SCALE"
CMD="$CMD --num_inference_steps $INFERENCE_STEPS"
CMD="$CMD --seed $SEED"
CMD="$CMD --device $DEVICE"
if [ -n "$RESIZE_INPUT" ]; then
    CMD="$CMD $RESIZE_INPUT"
fi
if [ -n "$SHARD_INDEX" ]; then
    CMD="$CMD --shard_index $SHARD_INDEX"
fi
if [ -n "$NUM_SHARDS" ]; then
    CMD="$CMD --num_shards $NUM_SHARDS"
fi

# Log file
if [ -n "$SHARD_INDEX" ]; then
    LOG_FILE="$OUTPUT_DIR/inference_shard${SHARD_INDEX}_$(date +%Y%m%d_%H%M%S).log"
else
    LOG_FILE="$OUTPUT_DIR/inference_$(date +%Y%m%d_%H%M%S).log"
fi

echo -e "${BLUE}Starting inference...${NC}"
echo "Command: $CMD"
echo "Log file: $LOG_FILE"
echo ""

# Run the inference
{
    echo "================================================================"
    echo "Physics-IQ WAN2.2 Inference Log"
    echo "Started at: $(date)"
    echo "================================================================"
    echo "Command: $CMD"
    echo "================================================================"
    echo ""
} > "$LOG_FILE"

# Execute with real-time output and logging
$CMD 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}✅ Inference completed successfully!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo -e "${BLUE}Output Summary:${NC}"
    echo "  Generated videos: $OUTPUT_DIR/.$MODEL_NAME/"
    echo "  Inference report: $OUTPUT_DIR/physics_iq_inference_report.md"
    echo "  Log file: $LOG_FILE"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Review generated videos in: $OUTPUT_DIR/.$MODEL_NAME/"
    echo "  2. Run Physics-IQ evaluation scripts on the output"
    echo "  3. Check the report for any failed generations"

    # Count generated videos
    if [ -d "$OUTPUT_DIR/.$MODEL_NAME" ]; then
        VIDEO_COUNT=$(find "$OUTPUT_DIR/.$MODEL_NAME" -name "*.mp4" | wc -l)
        echo ""
        echo -e "${GREEN}Generated $VIDEO_COUNT videos${NC}"
    fi
else
    echo ""
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}❌ Inference failed!${NC}"
    echo -e "${RED}================================================${NC}"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi

# Display timing information
echo ""
echo "Completed at: $(date)" | tee -a "$LOG_FILE"

# Optional: Display last few lines of the report
if [ -f "$OUTPUT_DIR/physics_iq_inference_report.md" ]; then
    echo ""
    echo -e "${BLUE}Report Summary:${NC}"
    tail -n 20 "$OUTPUT_DIR/physics_iq_inference_report.md"
fi