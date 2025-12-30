#!/bin/bash

# Physics-IQ Benchmark Inference Script for Wan2.2-TI2V-5B Stage 2 LoRA
# This script runs the Wan2.2-TI2V-5B model with Stage 2 LoRA weights

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Physics-IQ Benchmark Wan2.2-TI2V-5B Stage2 LoRA${NC}"
echo -e "${BLUE}================================================${NC}"

# Configuration
OUTPUT_DIR="physics_iq_results_stage2_long_step1000"
MODEL_NAME="wan22_ti2v_stage2_lora"
MAX_SAMPLES=""  # Leave empty for all samples, or set a number like "10"
NUM_FRAMES=81   # 81 frames at 16fps = ~5 seconds (benchmark requirement)
FPS=16          # Benchmark evaluates first 5 seconds
HEIGHT=480
WIDTH=832
CFG_SCALE=7.0
INFERENCE_STEPS=50
SEED=42
DEVICE="cuda"
WORLD_SIZE=4
MODEL_BASE_PATH="/nyx-storage1/hanliu/world_model_ckpt"
LORA_DIR="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage2_spatio_depth_full_long/checkpoint-1000"
LORA_CKPT=""

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
        --world-size)
            WORLD_SIZE="$2"
            shift 2
            ;;
        --model-base-path)
            MODEL_BASE_PATH="$2"
            shift 2
            ;;
        --lora-dir)
            LORA_DIR="$2"
            shift 2
            ;;
        --lora-checkpoint)
            LORA_CKPT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --max-samples N         Process only N samples (default: all)"
            echo "  --output-dir DIR        Output directory (default: physics_iq_results_stage2)"
            echo "  --model-name NAME       Model name for output folder (default: wan22_ti2v_stage2_lora)"
            echo "  --num-frames N          Number of frames to generate (default: 81)"
            echo "  --fps N                 Output video FPS (default: 16)"
            echo "  --height N              Video height (default: 480)"
            echo "  --width N               Video width (default: 832)"
            echo "  --cfg-scale N           CFG scale (default: 7.0)"
            echo "  --inference-steps N     Inference steps (default: 50)"
            echo "  --seed N                Random seed (default: 42)"
            echo "  --resize-input          Resize input images to output dimensions"
            echo "  --device DEVICE         Device to use (default: cuda)"
            echo "  --world-size N          Number of GPUs to use (default: 4)"
            echo "  --model-base-path DIR   Base WAN model path (default: /nyx-storage1/hanliu/world_model_ckpt)"
            echo "  --lora-dir DIR          Stage 2 final dir (default: /nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage2/final)"
            echo "  --lora-checkpoint FILE  Override LoRA checkpoint path"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run on first 10 samples"
            echo "  $0 --max-samples 10"
            echo ""
            echo "  # Run with custom output directory and model name"
            echo "  $0 --output-dir my_results --model-name wan22_stage2_custom"
            echo ""
            echo "  # Run with a custom Stage 2 directory"
            echo "  $0 --lora-dir /path/to/wan22_ti2v_stage2/final"
            echo ""
            echo "  # Multi-GPU: Launch 4 GPUs"
            echo "  $0 --world-size 4"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ -z "$LORA_CKPT" ]; then
    LORA_CKPT="${LORA_DIR}/lora_weights.pth"
fi

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model Name: $MODEL_NAME"
echo "  Model Base Path: $MODEL_BASE_PATH"
echo "  LoRA Checkpoint: $LORA_CKPT"
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
echo "  World Size: $WORLD_SIZE"
if [ -n "$RESIZE_INPUT" ]; then
    echo "  Resize Input: Yes"
else
    echo "  Resize Input: No"
fi
echo ""

# Check if Python script exists
SCRIPT_PATH="inference_physics_iq_wan22.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: $SCRIPT_PATH not found!${NC}"
    echo "Please ensure the inference script is in the current directory."
    exit 1
fi

# Check LoRA checkpoint
if [ ! -f "$LORA_CKPT" ]; then
    echo -e "${RED}Error: LoRA checkpoint not found: $LORA_CKPT${NC}"
    echo "Use --lora-dir or --lora-checkpoint to specify the Stage 2 weights."
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

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo -e "${BLUE}Starting multi-GPU inference...${NC}"
echo ""

for rank in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Launching GPU $rank..."

    CMD="python $SCRIPT_PATH \
        --output_dir $OUTPUT_DIR \
        --model_name $MODEL_NAME \
        --model_base_path $MODEL_BASE_PATH \
        --lora_checkpoint $LORA_CKPT \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --fps $FPS \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $INFERENCE_STEPS \
        --seed $SEED \
        --device $DEVICE \
        --shard_index $rank \
        --num_shards $WORLD_SIZE"

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    if [ -n "$RESIZE_INPUT" ]; then
        CMD="$CMD $RESIZE_INPUT"
    fi

    CUDA_VISIBLE_DEVICES=$rank $CMD > logs/stage2_long_gpu_${rank}.log 2>&1 &

    PID=$!
    echo "GPU $rank started with PID $PID (log: logs/stage2_long_gpu_${rank}.log)"
    sleep 2
done

echo ""
echo "================================================"
echo "All GPU processes launched!"
echo "================================================"
echo "Monitor progress with:"
echo "  tail -f logs/stage2_long_gpu_0.log"
echo "  tail -f logs/stage2_long_gpu_1.log"
echo "  tail -f logs/stage2_long_gpu_2.log"
echo "  tail -f logs/stage2_long_gpu_3.log"
echo ""
echo "Or monitor all at once:"
echo "  tail -f logs/stage2_long_gpu_*.log"
echo ""
echo "Check running processes:" 
echo "  ps aux | grep '$SCRIPT_PATH'"
echo ""
echo "To wait for all processes to complete:"
echo "  wait"
echo ""
echo "To kill all processes, run: pkill -f '$SCRIPT_PATH'"
