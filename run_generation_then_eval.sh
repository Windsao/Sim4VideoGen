#!/bin/bash

set -euo pipefail

GENERATION_SCRIPT="run_physics_iq_inference_stage2_layer_parallel.sh"
EVAL_SCRIPT="eval_iq.sh"
DEFAULT_OUTPUT_DIR="physics_iq_results_stage2_large_dataset_full_mse"
DEFAULT_MODEL_NAME="wan22_ti2v_stage2_layer"

if [ ! -f "$GENERATION_SCRIPT" ]; then
    echo "Error: $GENERATION_SCRIPT not found."
    exit 1
fi
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Error: $EVAL_SCRIPT not found."
    exit 1
fi

OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
MODEL_NAME="$DEFAULT_MODEL_NAME"
HELP_ONLY="false"
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --finetune-dir)
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        --help)
            HELP_ONLY="true"
            PASS_ARGS+=("$1")
            shift
            ;;
        *)
            PASS_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Running generation script: $GENERATION_SCRIPT"
bash "$GENERATION_SCRIPT" "${PASS_ARGS[@]}"

if [ "$HELP_ONLY" == "true" ]; then
    exit 0
fi

echo "Generation complete. Running eval script: $EVAL_SCRIPT"
bash "$EVAL_SCRIPT" --output-dir "$OUTPUT_DIR" --model-name "$MODEL_NAME"
