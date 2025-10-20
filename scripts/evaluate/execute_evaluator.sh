#!/bin/bash
# ==============================================================================
#                 Chart2Code Execute Execution Script
#
# This script automates the execution of generated Python scripts for evaluation.
# It iterates through specified source directories, finds all model-generated
# .py files, and passes them to the `execute_evaluator.py` script for processing.
#
# ==============================================================================

# Exit immediately on command failure.
set -e

# ==============================================================================
#                      Configuration Section
# ==============================================================================

# Get the script's directory for robust path handling.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Define the project's root directory (assumes this script is two levels deep).
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# 1. Path to the Python execution script.
PYTHON_SCRIPT_PATH="$PROJECT_ROOT/Evaluation/srcs/execute_evaluator.py"

# 2. Array of source directories containing model-specific result folders.
SOURCE_DIRS=(
    "$PROJECT_ROOT/Inference/level1_direct/results"
    "$PROJECT_ROOT/Inference/level1_customize/results"
    "$PROJECT_ROOT/Inference/level1_figure/results"
    "$PROJECT_ROOT/Inference/level2/results"
    "$PROJECT_ROOT/Inference/level3/results"
)

# 3. Base directory where all execution outputs will be saved.
BASE_OUTPUT_DIR="$PROJECT_ROOT/Evaluation/execute_results"


# ==============================================================================
#                           Main Script Logic
# ==============================================================================

echo "============================================="
echo "   Starting Automated Matplotlib Evaluation"
echo "============================================="
echo
echo "Project Root:      $PROJECT_ROOT"
echo "Python Processor:  $PYTHON_SCRIPT_PATH"
echo "Shared Output Dir: $BASE_OUTPUT_DIR"
echo

# Verify that the Python script exists before starting.
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python processor script not found at: $PYTHON_SCRIPT_PATH"
    exit 1
fi

# Ensure the main output directory exists.
mkdir -p "$BASE_OUTPUT_DIR"

# Loop through each top-level source directory.
for BASE_SOURCE_DIR in "${SOURCE_DIRS[@]}"; do
    echo -e "\n\n======================================================================="
    echo "=== Processing Top-Level Source: $BASE_SOURCE_DIR"
    echo "======================================================================="

    # Skip if the base directory doesn't exist.
    if [ ! -d "$BASE_SOURCE_DIR" ]; then
        echo "Warning: Base source directory not found, skipping: $BASE_SOURCE_DIR"
        continue
    fi

    # Find all subdirectories, excluding any ending in '_failed'.
    # Using `while read` ensures that directory names with spaces are handled correctly.
    find "$BASE_SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | grep -v '_failed$' | while IFS= read -r source_dir; do
        
        # Extract the short name of the directory (e.g., "qwen_level1_direct").
        dir_basename=$(basename "$source_dir")
        
        # Define a specific output path for this model's results.
        specific_output_dir="$BASE_OUTPUT_DIR/$dir_basename"

        echo "---------------------------------------------"
        echo "Processing Target: $dir_basename"
        echo "  -> Source: $source_dir"
        echo "  -> Output: $specific_output_dir"
        
        mkdir -p "$specific_output_dir"

        # Find all .py files in the source directory (non-recursively).
        # `mapfile` robustly handles filenames with spaces.
        mapfile -t script_files < <(find "$source_dir" -maxdepth 1 -type f -name "*.py")

        if [ ${#script_files[@]} -eq 0 ]; then
            echo "Warning: No .py files found in '$source_dir'. Skipping."
            echo
            continue
        fi

        echo "Found ${#script_files[@]} .py files. Calling Python script..."

        # Call the Python script, passing the output directory and all found .py files.
        python "$PYTHON_SCRIPT_PATH" \
            --output-dir "$specific_output_dir" \
            "${script_files[@]}"

        exit_code=$?

        echo "Python script finished for '$dir_basename' with exit code: $exit_code"
        
        if [ $exit_code -ne 0 ]; then
            echo "## WARNING: Python script exited with a non-zero status for '$dir_basename'."
        fi
        echo
    done
done

echo
echo "============================================="
echo "   All valid directories have been processed."
echo "============================================="