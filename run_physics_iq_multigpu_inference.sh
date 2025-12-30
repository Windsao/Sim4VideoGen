#!/bin/bash

# Multi-GPU Inference Launcher
# This script launches 4 separate inference processes, one per GPU
# Each process handles 25% of the prompts

# Configuration
SCRIPT="inference_physics_iq_wan22_multigpu.py"
WORLD_SIZE=4
OUTPUT_DIR="physics_iq_results"
MODEL_NAME="wan22_ti2v_5b_multigpu"

# Video generation parameters
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
CFG_SCALE=7.0
NUM_INFERENCE_STEPS=50
FPS=16
SEED=42

# Optional: limit number of samples for testing (comment out for full dataset)
# MAX_SAMPLES=40

echo "=========================================="
echo "Starting Multi-GPU Inference Pipeline"
echo "=========================================="
echo "World size: $WORLD_SIZE GPUs"
echo "Output directory: $OUTPUT_DIR"
echo "Model name: $MODEL_NAME"
echo "=========================================="
echo ""

# Create log directory
mkdir -p logs

# Launch processes for each GPU in the background
for rank in $(seq 0 $((WORLD_SIZE-1))); do
    echo "Launching GPU $rank..."

    # Build command with optional max_samples
    CMD="python $SCRIPT \
        --rank $rank \
        --world_size $WORLD_SIZE \
        --output_dir $OUTPUT_DIR \
        --model_name $MODEL_NAME \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --cfg_scale $CFG_SCALE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --fps $FPS \
        --seed $SEED \
        --resize_input"

    # Add max_samples if defined
    if [ ! -z "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi

    # Launch in background and redirect output to log file
    CUDA_VISIBLE_DEVICES=$rank $CMD > logs/gpu_${rank}.log 2>&1 &

    # Save process ID
    PID=$!
    echo "GPU $rank started with PID $PID (log: logs/gpu_${rank}.log)"

    # Small delay to stagger startup
    sleep 2
done

echo ""
echo "=========================================="
echo "All GPU processes launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/gpu_0.log"
echo "  tail -f logs/gpu_1.log"
echo "  tail -f logs/gpu_2.log"
echo "  tail -f logs/gpu_3.log"
echo ""
echo "Or monitor all at once:"
echo "  tail -f logs/gpu_*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep '$SCRIPT'"
echo ""
echo "Wait for all processes to complete:"
echo "  wait"
echo ""

# Optional: wait for all background jobs to complete
# Uncomment the next line if you want the script to wait
# wait

echo "To wait for completion, run: wait"
echo "To kill all processes, run: pkill -f '$SCRIPT'"
