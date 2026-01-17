#!/bin/bash

# Physics-IQ Benchmark Inference Script for CogVideoX-5b-I2V (diffusers)
export HF_HOME="/nyx-storage1/hanliu/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Physics-IQ CogVideoX-5b-I2V (diffusers)${NC}"
echo -e "${BLUE}================================================${NC}"

OUTPUT_DIR="physics_iq_results_cogvideox_i2v"
MODEL_NAME="cogvideox_i2v_5b"
MODEL_ID="THUDM/CogVideoX-5b-I2V"
MAX_SAMPLES=""
NUM_FRAMES=51
FPS=10
HEIGHT=480
WIDTH=832
CFG_SCALE=6.0
INFERENCE_STEPS=50
SEED=42
DEVICE="cuda"
WORLD_SIZE=4
HALF_MODE=""
ENABLE_OFFLOAD=""
VAE_TILING=""
VAE_SLICING=""

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
        --model-id)
            MODEL_ID="$2"
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
        --half)
            HALF_MODE="$2"
            shift 2
            ;;
        --enable-offload)
            ENABLE_OFFLOAD="--enable_offload"
            shift
            ;;
        --vae-tiling)
            VAE_TILING="--vae_tiling"
            shift
            ;;
        --vae-slicing)
            VAE_SLICING="--vae_slicing"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --max-samples N         Process only N samples (default: all)"
            echo "  --output-dir DIR        Output directory (default: physics_iq_results_cogvideox_i2v)"
            echo "  --model-name NAME       Model name for output folder (default: cogvideox_i2v_5b)"
            echo "  --model-id ID           HuggingFace model ID (default: THUDM/CogVideoX-5b-I2V)"
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
            echo "  --half first|second     Process first or second 50% of the dataset"
            echo "  --enable-offload        Enable sequential CPU offload"
            echo "  --vae-tiling            Enable VAE tiling"
            echo "  --vae-slicing           Enable VAE slicing"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuration:${NC}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model Name: $MODEL_NAME"
echo "  Model ID: $MODEL_ID"
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
if [ -n "$HALF_MODE" ]; then
    echo "  Half Mode: $HALF_MODE"
fi
if [ -n "$RESIZE_INPUT" ]; then
    echo "  Resize Input: Yes"
else
    echo "  Resize Input: No"
fi
if [ -n "$ENABLE_OFFLOAD" ]; then
    echo "  Offload: Enabled"
fi
if [ -n "$VAE_TILING" ]; then
    echo "  VAE Tiling: Enabled"
fi
if [ -n "$VAE_SLICING" ]; then
    echo "  VAE Slicing: Enabled"
fi
echo ""

SCRIPT_PATH="inference_physics_iq_cogvideox_i2v.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: $SCRIPT_PATH not found!${NC}"
    echo "Please ensure the inference script is in the current directory."
    exit 1
fi

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

mkdir -p "$OUTPUT_DIR"
mkdir -p logs_mse

echo -e "${BLUE}Starting multi-GPU inference...${NC}"
echo ""

PIDS=()
for rank in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Launching GPU $rank..."

    SHARD_INDEX=$rank
    NUM_SHARDS=$WORLD_SIZE
    if [ -n "$HALF_MODE" ]; then
        case "$HALF_MODE" in
            first|0)
                SHARD_INDEX=$rank
                ;;
            second|1)
                SHARD_INDEX=$((rank + WORLD_SIZE))
                ;;
            *)
                echo -e "${RED}Invalid --half value: $HALF_MODE (use first|second|0|1)${NC}"
                exit 1
                ;;
        esac
        NUM_SHARDS=$((WORLD_SIZE * 2))
    fi

    CMD="python $SCRIPT_PATH \
        --output_dir $OUTPUT_DIR \
        --model_name $MODEL_NAME \
        --model_id $MODEL_ID \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --fps $FPS \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $INFERENCE_STEPS \
        --seed $SEED \
        --device $DEVICE \
        --shard_index $SHARD_INDEX \
        --num_shards $NUM_SHARDS"

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    if [ -n "$RESIZE_INPUT" ]; then
        CMD="$CMD $RESIZE_INPUT"
    fi
    if [ -n "$ENABLE_OFFLOAD" ]; then
        CMD="$CMD $ENABLE_OFFLOAD"
    fi
    if [ -n "$VAE_TILING" ]; then
        CMD="$CMD $VAE_TILING"
    fi
    if [ -n "$VAE_SLICING" ]; then
        CMD="$CMD $VAE_SLICING"
    fi

    CUDA_VISIBLE_DEVICES=$rank $CMD > logs_mse/cogvideox_gpu_${rank}.log 2>&1 &

    PID=$!
    PIDS+=("$PID")
    echo "GPU $rank started with PID $PID (log: logs_mse/cogvideox_gpu_${rank}.log)"
    sleep 2
done

echo ""
echo "================================================"
echo "All GPU processes launched!"
echo "================================================"
echo "Monitor progress with:"
echo "  tail -f logs_mse/cogvideox_gpu_0.log"
echo "  tail -f logs_mse/cogvideox_gpu_1.log"
echo "  tail -f logs_mse/cogvideox_gpu_2.log"
echo "  tail -f logs_mse/cogvideox_gpu_3.log"
echo ""
echo "Or monitor all at once:"
echo "  tail -f logs_mse/cogvideox_gpu_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep '$SCRIPT_PATH'"
echo ""
echo "Waiting for all processes to complete..."
echo ""
echo "To kill all processes, run: pkill -f '$SCRIPT_PATH'"

for PID in "${PIDS[@]}"; do
    wait "$PID"
done

echo "All processes finished."
