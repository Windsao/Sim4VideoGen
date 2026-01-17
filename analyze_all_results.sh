#!/bin/bash

# Script to analyze all Physics-IQ CSV results in physics_iq_results_stage2_* folders
# Usage: bash analyze_all_results.sh

BASE_DIR="/home/mzh1800/DiffSynth-Studio"
SCRIPT_PATH="${BASE_DIR}/analyze_physics_iq_results.py"

# Check if the analysis script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Analysis script not found at $SCRIPT_PATH"
    exit 1
fi

echo "========================================================================"
echo "Physics-IQ Results Analysis - Batch Processing"
echo "========================================================================"
echo ""

# Counter for tracking progress
total_csv=0
processed_csv=0
failed_csv=0

# Find all directories matching the pattern
for results_dir in ${BASE_DIR}/physics_iq_results*/results; do
    if [ -d "$results_dir" ]; then
        echo "------------------------------------------------------------------------"
        echo "Processing directory: $results_dir"
        echo "------------------------------------------------------------------------"

        # Count CSV files in this directory (including hidden files)
        csv_count=$(find "$results_dir" -maxdepth 1 -name "*.csv" -type f | wc -l)

        if [ $csv_count -eq 0 ]; then
            echo "Warning: No CSV files found in $results_dir"
            echo ""
            continue
        fi

        echo "Found $csv_count CSV file(s) in this directory"
        echo ""

        # Process each CSV file (including hidden files starting with .)
        while IFS= read -r csv_file; do
            if [ -f "$csv_file" ]; then
                total_csv=$((total_csv + 1))
                csv_name=$(basename "$csv_file")

                echo "  Processing: $csv_name"

                # Run the analysis script
                if python3 "$SCRIPT_PATH" "$csv_file" --output_dir "$results_dir"; then
                    processed_csv=$((processed_csv + 1))
                    echo "  ✓ Successfully processed: $csv_name"
                else
                    failed_csv=$((failed_csv + 1))
                    echo "  ✗ Failed to process: $csv_name"
                fi
                echo ""
            fi
        done < <(find "$results_dir" -maxdepth 1 -name "*.csv" -type f)
    fi
done

echo "========================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "========================================================================"
echo "Total CSV files found:       $total_csv"
echo "Successfully processed:      $processed_csv"
echo "Failed:                      $failed_csv"
echo ""

if [ $total_csv -eq 0 ]; then
    echo "No CSV files found in any physics_iq_results_stage2_*/results directories"
    echo "Please check that the directories exist and contain CSV files."
else
    echo "Analysis results have been saved to their respective results directories."
fi

echo "========================================================================"
