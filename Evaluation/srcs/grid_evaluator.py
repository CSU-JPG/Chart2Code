# grid_evaluator.py 
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
class GridMetrics:
    """Stores the evaluation results for grid comparison."""
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
    
    # Create a buffer to swallow any print statements.
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
    """
    # Phase 1: Run in a separate process to check for hangs or fatal errors.
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

# --- Grid Evaluator ---
class GridEvaluator:
    """
    Evaluates the presence and state of grid lines in matplotlib figures.
    """
    def __init__(self) -> None:
        self.metrics = GridMetrics()

    def __call__(self, gen_fig: Optional[Figure], gt_fig: Optional[Figure]) -> GridMetrics:
        """
        Compares the grid lines of a generated figure against a ground-truth figure.
        """
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not get a valid Figure object for comparison."
            return self.metrics
        try:
            generation_grids = self._extract_grids_from_figure(gen_fig)
            gt_grids = self._extract_grids_from_figure(gt_fig)
            self._calculate_metrics(generation_grids, gt_grids)
        except Exception as e:
            logger.error(f"Error during grid evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_grids_from_figure(self, fig: Figure) -> List[Dict[str, bool]]:
        """
        Extracts grid visibility status (X and Y) from all axes in a Figure object.
        """
        grids = []
        for ax in fig.axes:
            # Check if any grid line for a given axis is visible.
            x_grid_visible = any(line.get_visible() for line in ax.get_xgridlines())
            y_grid_visible = any(line.get_visible() for line in ax.get_ygridlines())
            
            # Only record axes that have at least one grid enabled.
            if x_grid_visible or y_grid_visible:
                grids.append({
                    'x_grid_visible': x_grid_visible,
                    'y_grid_visible': y_grid_visible
                })
        return grids

    def _calculate_metrics(self, generation_grids: List[Dict], gt_grids: List[Dict]) -> None:
        """
        Calculates precision, recall, and F1 score by comparing the detected grids.
        """
        # Case 1: Both figures have no grids (perfect match).
        if not generation_grids and not gt_grids:
            self.metrics.precision = 1.0
            self.metrics.recall = 1.0
            self.metrics.f1 = 1.0
            return

        # Case 2: One has grids and the other doesn't (zero match).
        if not gt_grids or not generation_grids:
            self.metrics.precision = 0.0
            self.metrics.recall = 0.0
            self.metrics.f1 = 0.0
            return

        # Case 3: Both have grids; find the number of matching grid configurations.
        n_correct = 0
        gt_grids_copy = gt_grids.copy()
        
        for gen_grid in generation_grids:
            if gen_grid in gt_grids_copy:
                n_correct += 1
                # Remove the matched grid to handle duplicates correctly.
                gt_grids_copy.remove(gen_grid)
        
        self.metrics.precision = n_correct / len(generation_grids)
        self.metrics.recall = n_correct / len(gt_grids)
        
        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1 = 2 * self.metrics.precision * self.metrics.recall / (self.metrics.precision + self.metrics.recall)
        else:
            self.metrics.f1 = 0.0

# --- Main Workflow and Parallel Processing ---
def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path) -> Tuple[str, GridMetrics]:
    """
    Processes a single pair of generated and ground-truth scripts, returning the evaluation metrics.
    """
    logger.info(f"Processing: {file_name}...")
    evaluator = GridEvaluator()
    
    gen_fig, gen_err = execute_code_and_get_figure(str(generation_dir / file_name))
    gt_fig, gt_err = execute_code_and_get_figure(str(gt_dir / file_name))
    
    metrics = evaluator(gen_fig, gt_fig)
    
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

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None) -> Dict[str, GridMetrics]:
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
                all_results[file_name] = GridMetrics(status=ExecutionStatus.FAILED, error_message=str(e))
                
    if output_file:
        save_results_to_json(all_results, output_file)
        
    return all_results

def save_results_to_json(results: Dict[str, GridMetrics], output_file: str) -> None:
    """Saves the aggregated evaluation results to a JSON file."""
    json_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "evaluator": "GridEvaluator"
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
    print("Grid Evaluator")
    
    generation_dir = PROJECT_PATH / "generation_code"
    gt_dir = PROJECT_PATH / "gt_code"
    
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure 'generation_code' and 'gt_code' directories exist at: {PROJECT_PATH}")
        return
        
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "grid_evaluation_results.json"),
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


