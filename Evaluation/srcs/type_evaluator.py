# type_evaluator.py (V11.1 - Final Version with Output Suppression)
from typing import List, Tuple, Any, Dict, Optional
from dotenv import load_dotenv
import os
import sys
import numpy as np
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

# --- Core Dependencies & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parent.resolve()))
sys.path.insert(0, str(PROJECT_PATH))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh

# --- Status & Data Classes ---
class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout" # Added for clarity based on usage

@dataclass
class ChartTypeMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: str = ""

# --- Timeout-Protected & Output-Suppressed Code Executor ---
def _execute_code_runner(code_file_path: str) -> Tuple[bool, Optional[str]]:
    """Helper function to run code in an isolated, monitored process, suppressing all print outputs."""
    import runpy, matplotlib, io
    from contextlib import redirect_stdout
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Disable plot showing or saving
    plt.show = plt.savefig = lambda *args, **kwargs: None
    plt.close('all')
    
    # Buffer to capture and discard stdout
    output_buffer = io.StringIO()
    
    try:
        # Redirect stdout to the buffer
        with redirect_stdout(output_buffer):
            runpy.run_path(str(code_file_path), run_name='__main__')
        return True, None
    except Exception as e:
        return False, f"Error during code execution: {e}"
    finally:
        plt.close('all')

def execute_code_and_get_figure(code_file_path: str, timeout: int = 60) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """Executes a script in a subprocess with a timeout and suppresses output."""
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
    
    # Local execution to capture the figure object after a successful dry run.
    # Output is suppressed here as well to ensure no console spam.
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            runpy.run_path(str(code_file_path), run_name='__main__')
        
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return None, "Code executed successfully but did not generate any Figure"
        return plt.figure(fig_nums[-1]), None
    except Exception as e:
        return None, f"Error while capturing Figure object: {e}"
    finally:
        plt.close('all')

# --- High-Strictness Chart Type Evaluator ---
class ChartTypeEvaluator:
    def __init__(self) -> None:
        self.metrics = ChartTypeMetrics()

    def __call__(self, gen_fig: Optional[plt.Figure], gt_fig: Optional[plt.Figure]) -> ChartTypeMetrics:
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not get a valid Figure object"
            return self.metrics
        try:
            generation_chart_types = self._extract_chart_types_from_figure(gen_fig)
            gt_chart_types = self._extract_chart_types_from_figure(gt_fig)
            self._calculate_metrics(generation_chart_types, gt_chart_types)
        except Exception as e:
            logger.error(f"Error during chart type evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_chart_types_from_figure(self, fig: plt.Figure) -> Dict[str, int]:

        detected_types = set()

        for ax in fig.axes:

            if ax.containers:
                for container in ax.containers:
                    if isinstance(container, plt.matplotlib.container.BarContainer):
                        try:
                            is_string_label = any(isinstance(label.get_text(), str) and not label.get_text().replace('.', '', 1).isdigit() for label in ax.get_xticklabels())
                            if is_string_label:
                                detected_types.add('bar')
                            else:
                                detected_types.add('histogram')
                        except Exception:
                            detected_types.add('bar_or_hist') 
                        continue 

            if any(isinstance(artist, QuadMesh) for artist in ax.collections):
                detected_types.add('heatmap')

            has_poly = any(isinstance(artist, PolyCollection) for artist in ax.collections)
            has_lines = any(isinstance(artist, Line2D) for artist in ax.lines)
            if has_poly and has_lines:
                is_violin = False
                for coll in ax.collections:
                    if isinstance(coll, PolyCollection) and len(coll.get_paths()) > 0:
                        vertices = coll.get_paths()[0].vertices
                        if np.all(vertices[:, 1] > 0) and np.abs(np.min(vertices[:, 0])) - np.abs(np.max(vertices[:, 0])) < 1e-6:
                            is_violin = True
                            break
                if is_violin:
                    detected_types.add('violin')

            if any(isinstance(artist, Line2D) for artist in ax.lines) and not detected_types.intersection({'violin'}):
                detected_types.add('line')

            if any(isinstance(artist, PathCollection) for artist in ax.collections):
                detected_types.add('scatter')

            if any(isinstance(artist, Wedge) for artist in ax.patches):
                detected_types.add('pie')
                
            if any(isinstance(artist, Rectangle) for artist in ax.patches) and not detected_types.intersection({'bar', 'histogram'}):

                if len(ax.lines) > len(ax.patches) * 2: 
                    detected_types.add('boxplot')
                elif not detected_types.intersection({'pie'}): 

                    detected_types.add('bar')


            if any(isinstance(artist, plt.matplotlib.image.AxesImage) for artist in ax.images):

                if 'heatmap' not in detected_types:
                    detected_types.add('image')

        return {chart_type: 1 for chart_type in detected_types}

    def _calculate_metrics(self, generation_chart_types: Dict[str, int], gt_chart_types: Dict[str, int]) -> None:
        """Calculates strict precision, recall, and F1-score based on the sets of detected chart types."""
        if not generation_chart_types and not gt_chart_types:
            self.metrics.precision = 1.0; self.metrics.recall = 1.0; self.metrics.f1 = 1.0
            return
            
        gen_types_set = set(generation_chart_types.keys())
        gt_types_set = set(gt_chart_types.keys())

        # True Positives: Types present in both ground truth and generation
        n_correct = len(gen_types_set.intersection(gt_types_set))
        
        # Total number of types detected in the generated plot
        total_generated = len(gen_types_set)
        # Total number of types that should have been in the plot
        total_gt = len(gt_types_set)

        self.metrics.precision = n_correct / total_generated if total_generated > 0 else 1.0 if not gt_types_set else 0.0
        self.metrics.recall = n_correct / total_gt if total_gt > 0 else 1.0 if not gen_types_set else 0.0
        
        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1 = 2 * self.metrics.precision * self.metrics.recall / (self.metrics.precision + self.metrics.recall)
        else:
            self.metrics.f1 = 0.0

# --- Main Flow & Parallel Processing ---
def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path) -> Tuple[str, ChartTypeMetrics]:
    logger.info(f"[{file_name}] Starting processing...")
    evaluator = ChartTypeEvaluator()
    gen_fig, gen_err = execute_code_and_get_figure(str(generation_dir / file_name))
    gt_fig, gt_err = execute_code_and_get_figure(str(gt_dir / file_name))
    metrics = evaluator(gen_fig, gt_fig)
    if gen_err or gt_err:
        if metrics.status == ExecutionStatus.SUCCESS: metrics.status = ExecutionStatus.FAILED
        if "timed out" in str(gen_err) or "timed out" in str(gt_err): metrics.status = ExecutionStatus.TIMEOUT
        metrics.error_message = f"GenErr: {gen_err}; GtErr: {gt_err}"
        logger.warning(f"[{file_name}] Processing failed: {metrics.error_message}")
    else:
        logger.info(f"[{file_name}] Processing successful (P:{metrics.precision:.2f} R:{metrics.recall:.2f} F1:{metrics.f1:.2f})")
    return file_name, metrics

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None) -> Dict[str, ChartTypeMetrics]:
    generation_path = Path(generation_dir)
    gt_path = Path(gt_dir)
    common_files = sorted(list(set(f.name for f in generation_path.glob("*.py")) & set(f.name for f in gt_path.glob("*.py"))))
    if not common_files:
        logger.warning("No matching file pairs found"); return {}
    if num_workers is None: num_workers = os.cpu_count()
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
                all_results[file_name] = ChartTypeMetrics(status=ExecutionStatus.FAILED, error_message=str(e))
    if output_file:
        save_results_to_json(all_results, output_file, generation_dir, gt_dir)
    return all_results

def save_results_to_json(results: Dict[str, ChartTypeMetrics], output_file: str, generation_dir: str, gt_dir: str) -> None:
    json_data = {"evaluation_info": {"timestamp": datetime.now().isoformat(), "evaluator": "ChartTypeEvaluator_V11.1"}, "individual_results": []}
    for file_name, metrics in sorted(results.items()):
        json_data["individual_results"].append({
            "file": file_name, "status": metrics.status.value,
            "precision": round(metrics.precision, 4), "recall": round(metrics.recall, 4), "f1": round(metrics.f1, 4),
            "error_message": metrics.error_message
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to: {output_file}")

def main():
    print("=" * 60)
    print("Chart Type Evaluation (V11.1 - Final with Output Suppression):")
    generation_dir = PROJECT_PATH / "generation_code"
    gt_dir = PROJECT_PATH / "gt_code"
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure 'generation_code' and 'gt_code' directories exist at: {PROJECT_PATH}")
        return
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "type_evaluation_results_final.json"),
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
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

