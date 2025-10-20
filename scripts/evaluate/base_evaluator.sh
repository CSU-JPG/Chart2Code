#!/bin/bash
# ==============================================================================
#                 Chart2Code Base Evaluation Script
#
# Discovers generated code directories, matches them with ground-truth data,
# and runs the Python evaluator. It supports parallel execution and automatically
# finds the project root to be portable.
# ==============================================================================

# Exit immediately if a command fails, a variable is unset, or a pipe fails.
set -euo pipefail

# --- 1. Core Configuration ---
MAX_PARALLEL_TASKS=2
NUM_WORKERS_PER_EVAL_TASK=8

# --- 2. Dynamic Path Discovery ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Searches upwards from the script's location to find the project root,
# which must contain both 'Evaluation' and 'Inference' directories.
find_project_root() {
    local current_dir="$1"
    while [[ "$current_dir" != "/" && -n "$current_dir" ]]; do
        if [[ -d "$current_dir/Evaluation" && -d "$current_dir/Inference" ]]; then
            realpath "$current_dir"
            return 0
        fi
        current_dir=$(dirname "$current_dir")
    done
    return 1
}

# Find and set the project root directory; exit if not found.
PROJECT_ROOT_DIR=$(find_project_root "$SCRIPT_DIR")
if [ -z "$PROJECT_ROOT_DIR" ]; then
    echo "Error: Could not dynamically locate the project root directory." >&2
    exit 1
fi

# --- 3. Key Path Definitions (Relative to Dynamic Root) ---
EXECUTE_RESULTS_DIR="${PROJECT_ROOT_DIR}/Evaluation/execute_results"
GT_DATA_DIR="${PROJECT_ROOT_DIR}/data"
EVALUATION_RESULTS_DIR="${PROJECT_ROOT_DIR}/Evaluation/evaluation_results"
PYTHON_EVALUATOR_SCRIPT_FULLPATH="${PROJECT_ROOT_DIR}/Evaluation/srcs/base_evaluator.py"

# --- 4. Script Initialization ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG_DIR="${EVALUATION_RESULTS_DIR}/_base_main_log"
MAIN_LOG_FILE="${MAIN_LOG_DIR}/base_main_${TIMESTAMP}.log"
mkdir -p "$MAIN_LOG_DIR"
mkdir -p "$EVALUATION_RESULTS_DIR"

# Log all script output (stdout & stderr) to both the console and a file.
exec &> >(tee -a "$MAIN_LOG_FILE")

echo "=============================================================================="
echo "                   Chart2Code Automated Batch Evaluation "
echo "=============================================================================="
echo "Project Root Found: ${PROJECT_ROOT_DIR}"
echo "Scanning for results in: ${EXECUTE_RESULTS_DIR}"
echo "Final results will be saved in: ${EVALUATION_RESULTS_DIR}"
echo "Main log for this run: ${MAIN_LOG_FILE}"
echo "------------------------------------------------------------------------------"

# --- 5. Task Discovery and Matching Logic ---
declare -a TASKS_GEN_DIRS=()
declare -a TASKS_GT_PATHS=()
declare -a TASKS_OUTPUT_DIRS=()

echo "Discovering and matching evaluation tasks..."
for gen_dir in "${EXECUTE_RESULTS_DIR}"/*/; do
    if [ ! -d "$gen_dir" ]; then continue; fi
    gen_dir_path=$(realpath "${gen_dir}")
    dir_name=$(basename "${gen_dir_path}")
    gt_json_file=""

    # Match directory names to their corresponding ground-truth JSON files.
    if [[ "$dir_name" == *customize* ]]; then gt_json_file="level1_customize.json"
    elif [[ "$dir_name" == *direct* ]]; then gt_json_file="level1_direct.json"
    elif [[ "$dir_name" == *figure* ]]; then gt_json_file="level1_figure.json"
    elif [[ "$dir_name" == *level2* ]]; then gt_json_file="level2.json"
    elif [[ "$dir_name" == *level3* ]]; then gt_json_file="level3.json"
    fi

    # If a match is found, verify the GT file exists and add it to the task list.
    if [ -n "$gt_json_file" ]; then
        full_gt_path="${GT_DATA_DIR}/${gt_json_file}"
        if [ -f "$full_gt_path" ]; then
            output_dir="${EVALUATION_RESULTS_DIR}/${dir_name}"
            TASKS_GEN_DIRS+=("${gen_dir_path}")
            TASKS_GT_PATHS+=("${full_gt_path}")
            TASKS_OUTPUT_DIRS+=("${output_dir}")
            echo "  [MATCH] '${dir_name}' -> '${gt_json_file}'"
        else
            echo "  [WARNING] Match for '${dir_name}' found, but GT file missing: ${full_gt_path}"
        fi
    else
        echo "  [SKIP] No matching rule for directory: '${dir_name}'"
    fi
done

# --- 6. Parallel Execution ---
total_tasks=${#TASKS_GEN_DIRS[@]}
if [ "$total_tasks" -eq 0 ]; then
    echo "------------------------------------------------------------------------------"
    echo "No valid tasks found to evaluate. Exiting."
    exit 0
fi

echo "------------------------------------------------------------------------------"
echo "Found ${total_tasks} valid tasks. Starting execution..."
start_time=$(date +%s)

# Runs a single evaluation task by calling the main Python evaluator script.
run_evaluation_task() {
    local gen_dir="$1"
    local gt_json="$2"
    local output_dir="$3"
    local task_id="$4"
    local task_name=$(basename "$gen_dir")

    echo "$(date) [Task ${task_id}] START: ${task_name}"
    mkdir -p "${output_dir}"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_basename="base_results_${timestamp}"

    python "$PYTHON_EVALUATOR_SCRIPT_FULLPATH" \
        --gen-dir "${gen_dir}" \
        --gt-json "${gt_json}" \
        --output-dir "${output_dir}" \
        --output-basename "${output_basename}" \
        --workers "${NUM_WORKERS_PER_EVAL_TASK}"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$(date) [Task ${task_id}] SUCCESS: ${task_name}"
    else
        echo "$(date) [Task ${task_id}] FAILED (Exit Code: ${exit_code}): ${task_name}"
    fi
    return ${exit_code}
}

# Export function and variables to be available in subshells (for GNU Parallel).
export -f run_evaluation_task
export PYTHON_EVALUATOR_SCRIPT_FULLPATH
export NUM_WORKERS_PER_EVAL_TASK

# Use GNU Parallel for concurrent execution if available; otherwise, run sequentially.
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for execution (Max Jobs: ${MAX_PARALLEL_TASKS})."
    # --halt now,fail=1: Abort all jobs immediately if any single job fails.
    parallel -j "${MAX_PARALLEL_TASKS}" --halt now,fail=1 \
        run_evaluation_task {1} {2} {3} {#} \
        ::: "${TASKS_GEN_DIRS[@]}" \
        ::: "${TASKS_GT_PATHS[@]}" \
        ::: "${TASKS_OUTPUT_DIRS[@]}"
    ALL_TASKS_SUCCESS=$?
else
    echo "GNU Parallel not found. Running tasks sequentially..."
    ALL_TASKS_SUCCESS=0
    for i in "${!TASKS_GEN_DIRS[@]}"; do
        run_evaluation_task "${TASKS_GEN_DIRS[$i]}" "${TASKS_GT_PATHS[$i]}" "${TASKS_OUTPUT_DIRS[$i]}" "$((i+1))" || ALL_TASKS_SUCCESS=1
    done
fi

# --- 7. Summary Report ---
end_time=$(date +%s)
duration=$((end_time - start_time))
echo -e "\n=============================================================================="
if [ "$ALL_TASKS_SUCCESS" -eq 0 ]; then
    echo "              All evaluation tasks completed successfully!                 "
else
    echo "              Some evaluation tasks failed. Please check logs.             "
fi
echo "              Total time taken: $((duration / 60)) min $((duration % 60)) sec"
echo "              Results are stored in: ${EVALUATION_RESULTS_DIR}"
echo "=============================================================================="

exit $ALL_TASKS_SUCCESS