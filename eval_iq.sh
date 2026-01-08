#!/bin/bash

set -euo pipefail

# bash run_physics_iq_inference_stage2_layer_parallel.sh --world-size 4
PHYSICS_IQ_ROOT="/nyx-storage1/hanliu/physics-IQ-benchmark"
OUTPUT_DIR="physics_iq_results_stage2_large_dataset"
MODEL_NAME="wan22_ti2v_stage2_layer"
INPUT_FOLDER=""
DESCRIPTIONS_FILE="descriptions/descriptions.csv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --input-folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --descriptions-file)
            DESCRIPTIONS_FILE="$2"
            shift 2
            ;;
        --project-root)
            PHYSICS_IQ_ROOT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR         Output directory used for inference"
            echo "  --model-name NAME        Model name used for output folder"
            echo "  --input-folder DIR       Override input folder (defaults to OUTPUT_DIR/.MODEL_NAME)"
            echo "  --descriptions-file FILE Descriptions CSV path (relative to project root)"
            echo "  --project-root DIR       Physics-IQ benchmark repo root"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${SCRIPT_DIR}/${OUTPUT_DIR}"
fi

if [ -z "$INPUT_FOLDER" ]; then
    INPUT_FOLDER="${OUTPUT_DIR}/.${MODEL_NAME}"
elif [[ "$INPUT_FOLDER" != /* ]]; then
    INPUT_FOLDER="${SCRIPT_DIR}/${INPUT_FOLDER}"
fi

cd "$PHYSICS_IQ_ROOT"
python3 code/run_physics_iq.py \
    --input_folders "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_DIR" \
    --descriptions_file "$DESCRIPTIONS_FILE"
