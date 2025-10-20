import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from collections import Counter
import io
from contextlib import redirect_stdout
import runpy
import multiprocessing
from enum import Enum
import time

from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Sub-Evaluator Imports ---
try:
    from color_evaluator import ColorEvaluator, ColorMetrics, ExecutionStatus
    from grid_evaluator import GridEvaluator, GridMetrics
    from layout_evaluator import LayoutEvaluator, LayoutMetrics
    from legend_evaluator import LegendEvaluator, LegendMetrics
    from parameter_evaluator import ParameterEvaluator, ParameterMetrics
    from text_evaluator import TextEvaluator, TextMetrics
    from type_evaluator import ChartTypeEvaluator, ChartTypeMetrics
except ImportError as e:
    print(f"Failed to import a sub-evaluator: {e}")
    print("Please ensure all evaluator .py files are present.")
    sys.exit(1)

# --- Global Configuration ---
# Logger instance, configured in the main() function to write to a file.
logger = logging.getLogger("BaseEvaluator")
load_dotenv()
# Set a fallback project path; primarily, paths are provided via arguments.
PROJECT_PATH = Path(__file__).resolve().parents[4]

# --- Code Execution Utilities ---
def _execute_code_runner(code_file_path: str) -> Tuple[bool, Optional[str]]:
    """A sandboxed function to run a Python script, designed to be executed in a separate process."""
    import runpy, matplotlib, io, os
    from contextlib import redirect_stdout

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = plt.savefig = lambda *args, **kwargs: None
    plt.close('all')
    
    output_buffer = io.StringIO()
    script_path = Path(code_file_path)
    original_directory = os.getcwd()

    try:
        os.chdir(script_path.parent)
        with redirect_stdout(output_buffer):
            runpy.run_path(script_path.name, run_name='__main__')
        return True, None
    except Exception as e:
        return False, f"Error during code execution: {e}"
    finally:
        # Restore the original state.
        os.chdir(original_directory)
        plt.close('all')

def execute_code_and_get_figure(code_file_path: str, timeout: int = 60) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """
    Executes a Python script in a separate process with a timeout and captures the last generated matplotlib Figure.
    """
    # First, run the code in a subprocess to check for errors and timeouts without capturing the figure.
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_code_runner, code_file_path)
        try:
            success, error_msg = future.result(timeout=timeout)
            if not success:
                return None, error_msg
        except TimeoutError:
            return None, f"Code execution timed out (> {timeout} seconds)"
        except Exception as e:
            return None, f"Executor encountered an unknown error: {e}"


    plt.close('all')
    original_show = plt.show
    original_savefig = plt.savefig
    plt.show = lambda *args, **kwargs: None
    plt.savefig = lambda *args, **kwargs: None
    
    output_buffer = io.StringIO()
    script_path = Path(code_file_path)
    original_directory = os.getcwd()
    
    try:
        os.chdir(script_path.parent)
        with redirect_stdout(output_buffer):
            runpy.run_path(script_path.name, run_name='__main__')
        
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return None, "Code executed successfully but did not generate any Figure"
        
        # Return the last created figure.
        return plt.figure(fig_nums[-1]), None
    except Exception as e:
        return None, f"Error while capturing Figure object: {e}"
    finally:
        plt.show = original_show
        plt.savefig = original_savefig
        os.chdir(original_directory)
        plt.close('all')

# --- Serialization Helper ---
def convert_metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    """Recursively converts a metrics object to a JSON-serializable dictionary."""
    if not isinstance(metrics, object) or not hasattr(metrics, '__dict__'):
        return metrics
    
    result_dict = {}
    for key, value in metrics.__dict__.items():
        if isinstance(value, Enum):
            result_dict[key] = value.value
        elif hasattr(value, '__dict__'):
            result_dict[key] = convert_metrics_to_dict(value)
        else:
            result_dict[key] = value
    return result_dict

# --- Worker Process Initializer for Logging ---
def init_worker(log_file_path: str):
    """
    Initializes the logger for each worker process.
    This function is passed to the ProcessPoolExecutor's initializer.
    """
    worker_logger = logging.getLogger("BaseEvaluator")
    worker_logger.handlers = []
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)


# --- Parallel Evaluation Worker ---
def evaluate_and_save_single_file(
    file_name: str, 
    generation_dir: str, 
    gt_file_path: str,
    results_dir: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Orchestrates the full evaluation process for a single generated file against its ground truth.
    This function is designed to be run in a parallel process.
    """
    logger.info(f"Processing: {file_name}...")
    generation_file_path = str(Path(generation_dir) / file_name)
    
    gen_fig, gen_err = execute_code_and_get_figure(generation_file_path)
    gt_fig, gt_err = execute_code_and_get_figure(gt_file_path)
    
    # Determine if both scripts ran without errors.
    EXECUTION_SUCCESS = not (gen_err or gt_err)
    execution_error_msg = f"GenErr: {gen_err}; GtErr: {gt_err}"
    
    full_report = {}
    
    def run_evaluation(dim_name, evaluator_class, *args):
        """Helper to run a specific evaluator and handle execution failures gracefully."""
        evaluator_instance = evaluator_class()
        if EXECUTION_SUCCESS:
            metrics = evaluator_instance(*args)
            return convert_metrics_to_dict(metrics)
        else:
            # If code execution failed, return a standard failure metric object.
            metrics_class = globals()[evaluator_class.__name__.replace("Evaluator", "Metrics")]
            return convert_metrics_to_dict(metrics_class(status=ExecutionStatus.FAILED, error_message=execution_error_msg))

    # Run all dimension evaluators.
    full_report['color'] = run_evaluation('color', ColorEvaluator, gen_fig, gt_fig)
    full_report['layout'] = run_evaluation('layout', LayoutEvaluator, gen_fig, gt_fig, generation_file_path, gt_file_path)
    full_report['grid'] = run_evaluation('grid', GridEvaluator, gen_fig, gt_fig)
    full_report['legend'] = run_evaluation('legend', LegendEvaluator, gen_fig, gt_fig)
    full_report['parameter'] = run_evaluation('parameter', ParameterEvaluator, gen_fig, gt_fig)
    full_report['text'] = run_evaluation('text', TextEvaluator, gen_fig, gt_fig)
    full_report['type'] = run_evaluation('type', ChartTypeEvaluator, gen_fig, gt_fig)
    
    # Save the detailed individual report to a JSON file.
    output_path = Path(results_dir) / f"{Path(file_name).stem}_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
        
    # Prepare a concise summary for the final report.
    summary_data = {'file_name': file_name, 'is_success': EXECUTION_SUCCESS}
    if not EXECUTION_SUCCESS:
        summary_data['error_message'] = execution_error_msg
        
    for dim, result in full_report.items():
        if result.get('status') == 'success':
            if dim == 'parameter': # Special handling for parameter's nested metrics.
                summary_data[f'{dim}_data_f1'] = result.get('data_metrics', {}).get('f1', 0)
                summary_data[f'{dim}_visual_f1'] = result.get('visual_metrics', {}).get('f1', 0)
            else:
                summary_data[f'{dim}_f1'] = result.get('f1', 0)
    

    if EXECUTION_SUCCESS:
        logger.info(f"Finished: {file_name} (Success)")
    else:
        # Use logger.error to make it stand out and include the detailed reason.
        logger.error(f"Finished: {file_name} (Failed) | Reason: {execution_error_msg}")
        
    return file_name, summary_data

# --- Main Evaluator Class ---
class CodeEvaluator:
    """Handles the overall evaluation process, including file discovery, parallel execution, and reporting."""
    
    def evaluate_and_report(self, generation_dir: str, gt_json_path: str, output_dir: str, output_basename: str, log_file_path: str, num_workers: Optional[int] = None):
        """
        Main method to run the evaluation pipeline.
        """
        gen_path = Path(generation_dir)
        gt_json = Path(gt_json_path)
        output_path = Path(output_dir)

        # Discover common python files between the generation directory and the GT JSON manifest.
        if not gt_json.is_file():
            logger.error(f"Ground truth JSON file not found: {gt_json}")
            return
            
        with open(gt_json, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        # Create a map from basename (e.g., "3d_1.py") to its full path.
        gt_json_dir = gt_json.parent
        gt_files_map = {
            Path(item['GT code']).name: gt_json_dir / item['GT code']
            for item in gt_data if 'GT code' in item
        }
        
        gen_files = {f.name for f in gen_path.glob("*.py")}
        gt_basenames = set(gt_files_map.keys())
        common_files = sorted(list(gen_files & gt_basenames))
        
        if not common_files:
            logger.warning("No matching Python (.py) file pairs found between generation directory and GT JSON.")
            return
        if num_workers is None:
            num_workers = os.cpu_count()
        
        individual_results_dir = output_path / output_basename
        summary_report_path = output_path / f"{output_basename}.json"
        individual_results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Found {len(common_files)} file pairs to evaluate.")
        logger.info(f"Using {num_workers} workers for evaluation.")
        logger.info(f"Individual reports will be saved in: {individual_results_dir}")
        logger.info(f"Final summary report will be saved to: {summary_report_path}")

        all_summaries = []
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(log_file_path,)
        ) as executor:
            future_to_file = {
                executor.submit(
                    evaluate_and_save_single_file, 
                    fname, 
                    generation_dir, 
                    str(gt_files_map[fname]), 
                    str(individual_results_dir)
                ): fname 
                for fname in common_files
            }
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    _, summary_data = future.result()
                    all_summaries.append(summary_data)
                except Exception as e:
                    logger.error(f"A critical error occurred while processing {file_name}: {e}", exc_info=True)
                    all_summaries.append({'file_name': file_name, 'is_success': False, 'error_message': str(e)})

        self._generate_summary_report(all_summaries, str(summary_report_path), generation_dir, gt_json_path)

    def _generate_summary_report(self, all_summaries: List[Dict], output_path: str, gen_dir: str, gt_source: str):
        """Generates a final JSON summary report and logs a detailed, formatted summary."""
        if not all_summaries:
            logger.warning("No summary data was collected. Cannot generate report.")
            return

        successful_summaries = [s for s in all_summaries if s.get('is_success')]
        failed_summaries = [s for s in all_summaries if not s.get('is_success')]
        
        avg_scores = Counter()
        counts = Counter()
        for summary in successful_summaries:
            for key, value in summary.items():
                if key.endswith('_f1'):
                    avg_scores[key] += value if isinstance(value, (int, float)) else 0
                    counts[key] += 1
        
        for key in avg_scores:
            if counts[key] > 0:
                avg_scores[key] /= counts[key]
        
        total_files = len(all_summaries)
        success_count = len(successful_summaries)
        success_rate = round(success_count / total_files, 4) if total_files > 0 else 0
        
        # --- 1. Generate the JSON report file (this part is unchanged) ---
        report = {
            "evaluation_info": { 
                "timestamp": datetime.now().isoformat(), 
                "generation_directory": gen_dir, 
                "gt_source_file": gt_source, 
                "total_files_evaluated": total_files 
            },
            "success_rate": { "count": success_count, "total": total_files, "rate": success_rate },
            "average_f1_scores_on_success": {k: round(v, 4) for k, v in sorted(avg_scores.items())}
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Comprehensive summary JSON saved to: {output_path}")

        # --- 2. Log the detailed, formatted summary you wanted (ENHANCED SECTION) ---
        task_name = Path(gen_dir).name
        log_separator = "=" * 70
        
        logger.info(f"\n{log_separator}")
        logger.info(f"                EVALUATION SUMMARY FOR TASK: {task_name}")
        logger.info(f"{log_separator}")
        logger.info(f"Total Files Processed: {total_files}")
        logger.info(f"Code Execution Success Rate: {success_count} / {total_files} ({success_rate:.2%})")
        logger.info("-" * 70)
        
        if successful_summaries:
            logger.info("Average F1-Scores (calculated on successful executions only):")
            for dim, score in sorted(avg_scores.items()):
                logger.info(f"  - {dim:<25}: {score:.4f}")
        else:
            logger.info("No successful executions to calculate average F1-Scores.")
            
        if failed_summaries:
            logger.warning("-" * 70)
            logger.warning(f"Summary of {len(failed_summaries)} Failed Executions:")
            # To avoid flooding the log, limit the number of detailed errors shown.
            for i, s in enumerate(failed_summaries):
                if i < 10: # Show details for the first 10 failures
                    logger.warning(f"  - File: {s['file_name']} | Error: {s.get('error_message', 'Unknown error')}")
            if len(failed_summaries) > 10:
                logger.warning(f"  ... and {len(failed_summaries) - 10} more failures.")

        logger.info(f"{log_separator}\n")


def main():
    """Parses command-line arguments and initiates the evaluation process."""
    parser = argparse.ArgumentParser(description="A robust, multi-dimensional code evaluation framework.")
    parser.add_argument('--gen-dir', type=str, required=True, help="Directory containing generated code files.")
    parser.add_argument('--gt-json', type=str, required=True, help="Path to the ground-truth JSON file.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory where all results, logs, and reports will be saved.")
    parser.add_argument('--output-basename', type=str, required=True, help="Base name for all output files (e.g., 'base_results_TIMESTAMP').")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of parallel workers for file evaluation.")
    args = parser.parse_args()

    # Set up logging to both a file and the console.
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir_path / f"{args.output_basename}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s', 
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout) 
        ]
    )

    log_separator = "=" * 70
    logger.info(f"\n{log_separator}")
    logger.info(f"                STARTING NEW EVALUATION TASK")
    logger.info(f"{log_separator}")
    logger.info(f"Generation Directory : {args.gen_dir}")
    logger.info(f"Ground-Truth JSON    : {args.gt_json}")
    logger.info(f"Output Directory     : {args.output_dir}")
    logger.info(f"Output Basename      : {args.output_basename}")
    logger.info(f"Individual Log File  : {log_file_path}")
    logger.info(f"Parallel Workers     : {args.workers}")
    logger.info(f"{log_separator}")
    
    start_time = time.time()
    
    evaluator = CodeEvaluator()
    evaluator.evaluate_and_report(
        generation_dir=args.gen_dir,
        gt_json_path=args.gt_json,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        log_file_path=str(log_file_path),
        num_workers=args.workers
    )
    
    end_time = time.time()
    total_duration = end_time - start_time
    minutes = int(total_duration // 60)
    seconds = total_duration % 60
    logger.info(f"All evaluation tasks completed. Total time taken: {minutes} minutes {seconds:.2f} seconds.")

if __name__ == "__main__":
    # Set start method to 'spawn' for cross-platform consistency in multiprocessing.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # This can happen if the start method is already set.
        pass
    
    main()
