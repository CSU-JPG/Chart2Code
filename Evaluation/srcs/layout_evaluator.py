# layout_evaluator.py 
from typing import List, Tuple, Any, Dict, Optional
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import logging
import json
from datetime import datetime
import runpy
from enum import Enum
import io
from contextlib import redirect_stdout
from collections import Counter

# --- Core Dependencies and Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parent.resolve()))
sys.path.insert(0, str(PROJECT_PATH))

# Use a non-interactive backend for matplotlib to prevent GUI windows from appearing.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# --- Status and Data Classes ---
class ExecutionStatus(Enum):
    """Defines the possible outcomes of a script execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class LayoutMetrics:
    """Stores the evaluation results for layout comparison."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: str = ""

# --- Sandboxed Code Executor ---
def _execute_code_runner(code_file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Helper function to run a script in an isolated process, suppressing all standard output.
    This is designed to be the target for the ProcessPoolExecutor.
    """
    import runpy
    import matplotlib
    import io
    from contextlib import redirect_stdout

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Suppress plot rendering and close all figures to save memory.
    plt.show = lambda *args, **kwargs: None
    plt.savefig = lambda *args, **kwargs: None
    plt.close('all')
    
    output_buffer = io.StringIO()
    try:
        # Redirect all standard output to the buffer during execution.
        with redirect_stdout(output_buffer):
            runpy.run_path(str(code_file_path), run_name='__main__')
        return True, None
    except Exception as e:
        return False, f"Error during code execution: {e}"
    finally:
        plt.close('all')

def execute_code_and_get_figure(code_file_path: str, timeout: int = 60) -> Tuple[Optional[Figure], Optional[str]]:
    """
    Executes a plotting script with a timeout and returns the generated matplotlib figure.
    All standard output from the script is suppressed.
    """
    # Phase 1: Run in a separate process to safely handle timeouts and fatal errors.
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
    
    # Phase 2: If the first run was safe, run again locally to capture the figure.
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            runpy.run_path(str(code_file_path), run_name='__main__')
        
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return None, "Code executed successfully but did not generate any Figure"
        
        # Return the most recently created figure.
        return plt.figure(fig_nums[-1]), None
    except Exception as e:
        return None, f"Error while capturing Figure object: {e}"
    finally:
        plt.close('all')

# --- Layout Evaluator ---
class LayoutEvaluator:
    """
    Evaluates the subplot grid layout of matplotlib figures.
    """
    def __init__(self) -> None:
        self.metrics = LayoutMetrics()

    def __call__(self, gen_fig: Optional[Figure], gt_fig: Optional[Figure], gen_file_path: str, gt_file_path: str) -> LayoutMetrics:
        """
        Compares the layout of a generated figure against a ground-truth figure.
        """
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not get a valid Figure object for comparison."
            return self.metrics
        try:
            generation_layouts = self._extract_layout_from_figure(gen_fig, gen_file_path)
            gt_layouts = self._extract_layout_from_figure(gt_fig, gt_file_path)
            self._calculate_metrics(generation_layouts, gt_layouts)
        except Exception as e:
            logger.error(f"Error during layout evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_layout_from_figure(self, fig: Figure, file_path: str) -> List[Dict[str, int]]:
        """
        Extracts subplot grid layout information from a Figure object.
        It inspects the SubplotSpec of each axis to determine the grid geometry.
        """
        # Special handling for graph plots which may not use a standard GridSpec.
        if "/graph" in file_path:
            return [dict(nrows=1, ncols=1, row_start=0, row_end=0, col_start=0, col_end=0)]
        
        layout_info = []
        for ax in fig.axes:
            spec = ax.get_subplotspec()
            if spec is None: continue

            gs = spec.get_gridspec()
            nrows, ncols = gs.get_geometry()
            row_start, row_end = spec.rowspan.start, spec.rowspan.stop - 1
            col_start, col_end = spec.colspan.start, spec.colspan.stop - 1
            
            layout_info.append(dict(
                nrows=nrows, ncols=ncols, 
                row_start=row_start, row_end=row_end, 
                col_start=col_start, col_end=col_end
            ))
        return layout_info

    def _calculate_metrics(self, generation_layouts: List[Dict], gt_layouts: List[Dict]) -> None:
        """
        Calculates precision, recall, and F1 score by comparing layout configurations.
        """
        if not generation_layouts and not gt_layouts:
            self.metrics.precision = 1.0; self.metrics.recall = 1.0; self.metrics.f1 = 1.0
            return
        
        if not gt_layouts or not generation_layouts:
            self.metrics.precision = 0.0; self.metrics.recall = 0.0; self.metrics.f1 = 0.0
            return

        n_correct = 0
        gt_layouts_copy = gt_layouts.copy()
        for layout in generation_layouts:
            if layout in gt_layouts_copy:
                n_correct += 1
                # Remove the matched layout to handle duplicates correctly.
                gt_layouts_copy.remove(layout)
        
        self.metrics.precision = n_correct / len(generation_layouts) if generation_layouts else 1.0
        self.metrics.recall = n_correct / len(gt_layouts) if gt_layouts else 1.0
        
        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1 = 2 * self.metrics.precision * self.metrics.recall / (self.metrics.precision + self.metrics.recall)
        else:
            self.metrics.f1 = 0.0

# --- Main Workflow and Parallel Processing ---
def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path) -> Tuple[str, LayoutMetrics]:
    """
    Processes a single pair of generated and ground-truth scripts.
    """
    logger.info(f"Processing: {file_name}...")
    evaluator = LayoutEvaluator()
    generation_file = generation_dir / file_name
    gt_file = gt_dir / file_name
    
    gen_fig, gen_err = execute_code_and_get_figure(str(generation_file))
    gt_fig, gt_err = execute_code_and_get_figure(str(gt_file))
    
    metrics = evaluator(gen_fig, gt_fig, str(generation_file), str(gt_file))
    
    # Consolidate execution errors with evaluation metrics.
    if gen_err or gt_err:
        if metrics.status == ExecutionStatus.SUCCESS:
            metrics.status = ExecutionStatus.FAILED
        if "timed out" in str(gen_err).lower() or "timed out" in str(gt_err).lower():
            metrics.status = ExecutionStatus.TIMEOUT
        metrics.error_message = f"GenErr: {gen_err}; GtErr: {gt_err}"
        logger.warning(f"Failed to process {file_name}: {metrics.error_message}")
    else:
        logger.info(f"Finished {file_name} (P:{metrics.precision:.2f} R:{metrics.recall:.2f} F1:{metrics.f1:.2f})")
        
    return file_name, metrics

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None) -> Dict[str, LayoutMetrics]:
    """
    Evaluates all matching Python scripts in two directories in parallel.
    """
    generation_path = Path(generation_dir)
    gt_path = Path(gt_dir)
    
    common_files = sorted(list(set(f.name for f in generation_path.glob("*.py")) & set(f.name for f in gt_path.glob("*.py"))))
    
    if not common_files:
        logger.warning("No matching file pairs found in the specified directories."); return {}
        
    if num_workers is None:
        num_workers = os.cpu_count()
        
    logger.info(f"Found {len(common_files)} file pairs to process using {num_workers} workers.")
    
    all_results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_single_file, fname, generation_path, gt_path): fname for fname in common_files}
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                _, metrics = future.result()
                all_results[file_name] = metrics
            except Exception as e:
                logger.error(f"A critical error occurred while processing the future for {file_name}: {e}")
                all_results[file_name] = LayoutMetrics(status=ExecutionStatus.FAILED, error_message=str(e))
                
    if output_file:
        save_results_to_json(all_results, output_file)
        
    return all_results

def save_results_to_json(results: Dict[str, LayoutMetrics], output_file: str) -> None:
    """Saves the aggregated evaluation results to a JSON file."""
    json_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "evaluator": "LayoutEvaluator"
        },
        "individual_results": []
    }
    
    for file_name, metrics in sorted(results.items()):
        json_data["individual_results"].append({
            "file": file_name,
            "status": metrics.status.value,
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "error_message": metrics.error_message
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Evaluation results saved to: {output_file}")

def main():
    """Main entry point for the script."""
    print("=" * 60)
    print("Layout Evaluator")
    
    generation_dir = PROJECT_PATH / "generation_code"
    gt_dir = PROJECT_PATH / "gt_code"
    
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure 'generation_code' and 'gt_code' directories exist at: {PROJECT_PATH}")
        return
        
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "layout_evaluation_results.json"),
            num_workers=os.cpu_count()
        )
        
        if results:
            print("\n" + "=" * 25 + " Evaluation Summary " + "=" * 25)
            status_counts = Counter(m.status.value for m in results.values())
            total = len(results)
            print(f"Total files evaluated: {total}")
            for status, count in status_counts.items():
                print(f"  - {status.capitalize():<10}: {count:4d} files ({count/total:.1%})")
            print("=" * 60)
        else:
            print("\nNo files were evaluated.")
            
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()


