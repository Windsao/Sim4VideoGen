#!/bin/bash

# Simple 4-GPU PhyGenBench Launcher
# This script runs the inference script 4 times in parallel, each on a different GPU
# Each instance processes 25% of the prompts (961 total / 4 = ~240 per GPU)

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
NUM_GPUS=4
SCRIPT="./run_phygenbench_inference.sh"

# Pass through all arguments to each instance
EXTRA_ARGS="$@"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PhyGenBench 4-GPU Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "This will launch 4 parallel processes:"
echo "  GPU 0: Processing shard 0/4 (samples 0-39)"
echo "  GPU 1: Processing shard 1/4 (samples 40-79)"
echo "  GPU 2: Processing shard 2/4 (samples 80-119)"
echo "  GPU 3: Processing shard 3/4 (samples 120-159)"
echo ""

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo -e "${RED}Error: $SCRIPT not found!${NC}"
    exit 1
fi

# Check if prompts file exists
if [ ! -f "phygenbench_prompts.json" ]; then
    echo -e "${RED}Error: phygenbench_prompts.json not found!${NC}"
    exit 1
fi

echo -e "${YELLOW}Extra arguments: $EXTRA_ARGS${NC}"
echo ""

# Launch 4 processes, one per GPU
for gpu in $(seq 0 $((NUM_GPUS-1))); do
    echo -e "${GREEN}Launching GPU $gpu...${NC}"

    # Run in background with CUDA_VISIBLE_DEVICES set to specific GPU
    CUDA_VISIBLE_DEVICES=$gpu $SCRIPT --shard-index $gpu --num-shards $NUM_GPUS $EXTRA_ARGS &

    PID=$!
    echo "  → PID: $PID"

    # Small delay to stagger startup
    sleep 2
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All 4 GPU processes launched!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Monitor progress:"
echo "  - Check GPU usage: watch -n 1 nvidia-smi"
echo "  - View logs: tail -f phygenbench_results/inference_shard*.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep inference_phygenbench_wan22.py"
echo ""
echo "To kill all processes:"
echo "  pkill -f inference_phygenbench_wan22.py"
echo ""
echo "Waiting for all processes to complete..."
echo "(Press Ctrl+C to stop waiting, but processes will continue running)"
echo ""

# Wait for all background processes
wait

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All processes completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To merge reports from all shards:"
echo "  python merge_shard_reports.py --output-dir phygenbench_results --prefix phygenbench"
echo ""
