#!/bin/bash

# ===================================================================================
#                              User Configuration
# ===================================================================================
# This script is a generic runner for Python inference scripts.
# To run a different model, simply change the MODEL_IDENTIFIER variable below.

# 1. Model Identifier to Run.
MODEL_IDENTIFIER="qwen2.5_level3_7B"
#    Examples: "qwen2.5_level3_7B", "qwen2.5_level3_72B", "InternVL_2.5_level3_8B", etc.

# 2. Select the model loading mode: "hub" or "local".
LOAD_SOURCE="local"

# 3. Input JSON data file.
JSON_FILENAME="level3.json"

# 4. GPU Configuration.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

# ===================================================================================
#      Script Logic (Derived from Configuration - Do not modify)
# ===================================================================================
# Exit immediately if any command fails.
set -e

# --- Dynamically set names based on the model identifier ---
PYTHON_SCRIPT_NAME="Inference/level3/srcs/${MODEL_IDENTIFIER}.py"
RESULTS_RUN_NAME="${MODEL_IDENTIFIER}"
LOG_FILENAME="${MODEL_IDENTIFIER}.csv"

# --- Determine Project Root and Key directories ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../" &> /dev/null && pwd)

# --- Define all necessary absolute paths ---
PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/${PYTHON_SCRIPT_NAME}"
JSON_FILE_PATH="${PROJECT_ROOT}/data/${JSON_FILENAME}"

OUTPUT_BASE_DIR="${PROJECT_ROOT}/Inference/level3"
OUTPUT_DIR_PATH="${OUTPUT_BASE_DIR}/results/${RESULTS_RUN_NAME}"
LOG_DIR_PATH="${OUTPUT_BASE_DIR}/logs"
LOG_FILE_PATH="${LOG_DIR_PATH}/${LOG_FILENAME}"

# --- Pre-run Sanity Checks ---
echo "-----------------------------------------------------"
echo "Project Root:   ${PROJECT_ROOT}"
echo "Model ID:       ${MODEL_IDENTIFIER}"
echo "Python Script:  ${PYTHON_SCRIPT_PATH}"
echo "Load Source:    ${LOAD_SOURCE}"
echo "JSON Input:     ${JSON_FILE_PATH}"
echo "Output Dir:     ${OUTPUT_DIR_PATH}"
echo "Log File:       ${LOG_FILE_PATH}"
echo "Visible GPUs:   ${CUDA_VISIBLE_DEVICES}"
echo "-----------------------------------------------------"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at: $PYTHON_SCRIPT_PATH"
    echo "Please check if the model identifier '$MODEL_IDENTIFIER' is correct and the corresponding .py file exists."
    exit 1
fi

# --- Prepare Directories and Execute ---
echo "Preparing output directories..."
mkdir -p "${OUTPUT_DIR_PATH}"
mkdir -p "${LOG_DIR_PATH}"
echo "Output directories are ready."

echo "Starting Python inference script..."

python "${PYTHON_SCRIPT_PATH}" \
  --load_source "${LOAD_SOURCE}" \
  --json_path "${JSON_FILE_PATH}" \
  --output_dir "${OUTPUT_DIR_PATH}" \
  --log_path "${LOG_FILE_PATH}"

echo "Script execution finished for ${MODEL_IDENTIFIER}. 🚀"


