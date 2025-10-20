#!/bin/bash
# ==============================================================================
#                Chart2Code LLM Evaluation Script
#
# Discovers generated code, matches it with ground-truth data, and runs an
# LLM-based evaluator. Supports parallel execution via GNU Parallel.
# ==============================================================================

# Exit immediately on command failure, unset variable, or pipe failure.
set -euo pipefail

# --- 1. Core Configuration ---

# API model for evaluation (e.g., 'gpt-4o', 'gpt-4-turbo').
API_MODEL_NAME="gpt-5-mini"

# Max concurrent evaluation tasks.
MAX_PARALLEL_TASKS=1

# Number of workers per Python evaluation task.
NUM_WORKERS_PER_EVAL_TASK=4


# --- 2. Path Setup ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assumes this script is located in project_root/Evaluation/scripts
PROJECT_ROOT_DIR=$(realpath "${SCRIPT_DIR}/../..")

# --- Key Paths ---
PYTHON_EVALUATOR_SCRIPT_FULLPATH="${PROJECT_ROOT_DIR}/Evaluation/srcs/LLM_evaluator.py"
EXECUTE_RESULTS_DIR="${PROJECT_ROOT_DIR}/Evaluation/execute_results"
GT_JSONS_DIR="${PROJECT_ROOT_DIR}/data"
EVALUATION_RESULTS_DIR="${PROJECT_ROOT_DIR}/Evaluation/evaluation_results"


# --- 3. Script Initialization ---
mkdir -p "$EVALUATION_RESULTS_DIR"

echo "=============================================================================="
echo "              Automated Batch Code Similarity Evaluation"
echo "=============================================================================="
echo "Project Root: ${PROJECT_ROOT_DIR}"
echo "Evaluation Model: ${API_MODEL_NAME}"
echo "Results will be saved in: ${EVALUATION_RESULTS_DIR}"
echo "------------------------------------------------------------------------------"


# --- 4. Task Discovery and Matching ---
declare -a TASKS_GEN_DIRS=()
declare -a TASKS_GT_PATHS=()
declare -a TASKS_OUTPUT_DIRS=()

echo "Discovering and matching evaluation tasks..."
for gen_dir in "${EXECUTE_RESULTS_DIR}"/*/; do
    [ -d "${gen_dir}" ] || continue # Skip non-directory files
    gen_dir_path=$(realpath "${gen_dir}")
    dir_name=$(basename "${gen_dir_path}")
    gt_json_file=""

    # Match directory name to a ground-truth (GT) file.
    if [[ "$dir_name" == *customize* ]]; then gt_json_file="level1_customize.json"
    elif [[ "$dir_name" == *direct* ]]; then gt_json_file="level1_direct.json"
    elif [[ "$dir_name" == *figure* ]]; then gt_json_file="level1_figure.json"
    elif [[ "$dir_name" == *level2* ]]; then gt_json_file="level2.json"
    elif [[ "$dir_name" == *level3* ]]; then gt_json_file="level3.json"
    fi

    # If a match is found, verify the GT file exists and add it to the task list.
    if [ -n "$gt_json_file" ]; then
        full_gt_path="${GT_JSONS_DIR}/${gt_json_file}"
        if [ -f "$full_gt_path" ]; then
            output_dir="${EVALUATION_RESULTS_DIR}/${dir_name}"
            TASKS_GEN_DIRS+=("${gen_dir_path}")
            TASKS_GT_PATHS+=("${full_gt_path}")
            TASKS_OUTPUT_DIRS+=("${output_dir}")
            echo "  [MATCH] '${dir_name}' -> '${gt_json_file}'"
        else
            echo "  [WARNING] Match for '${dir_name}', but GT file not found: ${full_gt_path}"
        fi
    else
        echo "  [SKIP] No matching GT rule for directory: '${dir_name}'"
    fi
done


# --- 5. Execution ---
total_tasks=${#TASKS_GEN_DIRS[@]}
if [ "$total_tasks" -eq 0 ]; then
    echo "------------------------------------------------------------------------------"
    echo "No valid tasks found to evaluate. Exiting."
    exit 0
fi

echo "------------------------------------------------------------------------------"
echo "Found ${total_tasks} valid tasks. Starting execution..."
start_time=$(date +%s)

# Executes a single evaluation task, handling logging and output paths.
run_evaluation_task() {
    local gen_dir="$1"
    local gt_json="$2"
    local output_dir="$3"
    local task_id="$4"
    local task_name=$(basename "$gen_dir")

    echo "--- [Task ${task_id}/${total_tasks}] START: ${task_name} ---"
    mkdir -p "${output_dir}"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local base_name="LLM_results_${timestamp}"
    local log_file="${output_dir}/${base_name}.log"
    local summary_json_file="${output_dir}/${base_name}.json"
    local details_dir="${output_dir}/${base_name}"

    echo "Task outputs for '${task_name}' will use base name: ${base_name}"

    # Execute the python script and redirect all output to a task-specific log file.
    python3 "$PYTHON_EVALUATOR_SCRIPT_FULLPATH" \
        --gen-dir "${gen_dir}" \
        --gt-json "${gt_json}" \
        --summary-json-path "${summary_json_file}" \
        --details-dir "${details_dir}" \
        --model-name "${API_MODEL_NAME}" \
        --workers "${NUM_WORKERS_PER_EVAL_TASK}" > >(tee -a "${log_file}") 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "--- [Task ${task_id}/${total_tasks}] SUCCESS: ${task_name} ---"
    else
        echo "--- [Task ${task_id}/${total_tasks}] FAILED (Exit Code: ${exit_code}): ${task_name} ---"
    fi
    return ${exit_code}
}

# Export function and variables to be available in subshells (for GNU Parallel).
export -f run_evaluation_task
export PYTHON_EVALUATOR_SCRIPT_FULLPATH
export NUM_WORKERS_PER_EVAL_TASK
export API_MODEL_NAME
export total_tasks

# Use GNU Parallel if available; otherwise, run sequentially.
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for execution (Max Jobs: ${MAX_PARALLEL_TASKS})."
    # --halt now,fail=1: Aborts all jobs immediately if any single job fails.
    parallel -j "${MAX_PARALLEL_TASKS}" --halt now,fail=1 \
        run_evaluation_task {1} {2} {3} {#} \
        ::: "${TASKS_GEN_DIRS[@]}" \
        ::: "${TASKS_GT_PATHS[@]}" \
        ::: "${TASKS_OUTPUT_DIRS[@]}"
    ALL_TASKS_EXIT_CODE=$?
else
    echo "GNU Parallel not found. Running tasks sequentially..."
    ALL_TASKS_EXIT_CODE=0
    for i in "${!TASKS_GEN_DIRS[@]}"; do
        run_evaluation_task "${TASKS_GEN_DIRS[$i]}" "${TASKS_GT_PATHS[$i]}" "${TASKS_OUTPUT_DIRS[$i]}" "$((i+1))" || ALL_TASKS_EXIT_CODE=1
    done
fi


# --- 6. Summary Report ---
end_time=$(date +%s)
duration=$((end_time - start_time))

echo
echo "=============================================================================="
if [ "$ALL_TASKS_EXIT_CODE" -eq 0 ]; then
    echo "✅ All evaluation tasks completed successfully!"
else
    echo "⚠️  Some evaluation tasks failed. Check logs in the respective output directories."
fi
echo "    Total time taken: $((duration / 60)) min $((duration % 60)) sec"
echo "    All results are stored in: ${EVALUATION_RESULTS_DIR}"
echo "=============================================================================="
exit $ALL_TASKS_EXIT_CODE