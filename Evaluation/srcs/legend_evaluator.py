# legend_evaluation.py 
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

# --- Core Dependencies & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parent.resolve()))
sys.path.insert(0, str(PROJECT_PATH))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Status & Data Classes ---
class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class LegendMetrics:
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
    
    # Suppress figure rendering/saving within the subprocess
    plt.show = plt.savefig = lambda *args, **kwargs: None
    plt.close('all')
    
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            runpy.run_path(str(code_file_path), run_name='__main__')
        return True, None
    except Exception as e:
        return False, f"Error during code execution: {e}"
    finally:
        plt.close('all')

def execute_code_and_get_figure(code_file_path: str, timeout: int = 60) -> Tuple[Optional[plt.Figure], Optional[str]]:
    """Executes code with a timeout and suppresses output by using a temporary single-task process pool."""
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
    
    # Also suppress output for the local execution that captures the figure.
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

# --- Legend Evaluator Class ---
class LegendEvaluator:
    def __init__(self, use_position: bool = True) -> None:
        self.metrics = LegendMetrics()
        self.use_position = use_position

    def __call__(self, gen_fig: Optional[plt.Figure], gt_fig: Optional[plt.Figure]) -> LegendMetrics:
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not get a valid Figure object"
            return self.metrics
        try:
            gen_fig.canvas.draw()
            gt_fig.canvas.draw()
            generation_legends = self._extract_legends_from_figure(gen_fig)
            gt_legends = self._extract_legends_from_figure(gt_fig)
            self._calculate_metrics(generation_legends, gt_legends)
        except Exception as e:
            logger.error(f"Error during legend evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_legends_from_figure(self, fig: plt.Figure) -> List[Dict]:
        """Extracts text and bounding box information from all visible legends in a figure."""
        legends_info = []
        renderer = fig.canvas.get_renderer()
        all_legends = fig.legends[:]
        for ax in fig.axes:
            if ax.get_legend():
                all_legends.append(ax.get_legend())
        
        for legend in set(all_legends):
            if not legend or not legend.get_visible():
                continue
            
            legend_bbox = legend.get_window_extent(renderer)
            for text_obj in legend.get_texts():
                if text_obj.get_visible() and text_obj.get_text():
                    legends_info.append({
                        "text": text_obj.get_text(),
                        "bbox": (legend_bbox.x0, legend_bbox.y0, legend_bbox.x1, legend_bbox.y1)
                    })
        return legends_info

    def _calculate_metrics(self, generation_legends: List[Dict], gt_legends: List[Dict]) -> None:
        """Calculates precision, recall, and F1 score for legends."""
        if not generation_legends and not gt_legends:
            self.metrics.precision = 1.0; self.metrics.recall = 1.0; self.metrics.f1 = 1.0
            return
        
        if not gt_legends or not generation_legends:
            self.metrics.precision = 0.0; self.metrics.recall = 0.0; self.metrics.f1 = 0.0
            return
        
        n_correct = 0
        gt_legends_copy = gt_legends.copy()
        for gen_legend in generation_legends:
            best_match = None
            for gt_legend in gt_legends_copy:
                if gen_legend["text"] == gt_legend["text"]:
                    if self.use_position:
                        # Check for bounding box intersection (IoU > 0)
                        gen_box, gt_box = gen_legend["bbox"], gt_legend["bbox"]
                        xA = max(gen_box[0], gt_box[0]); yA = max(gen_box[1], gt_box[1])
                        xB = min(gen_box[2], gt_box[2]); yB = min(gen_box[3], gt_box[3])
                        interArea = max(0, xB - xA) * max(0, yB - yA)
                        if interArea > 0:
                            best_match = gt_legend
                            break
                    else:
                        best_match = gt_legend
                        break
            
            if best_match:
                n_correct += 1
                gt_legends_copy.remove(best_match)
        
        self.metrics.precision = n_correct / len(generation_legends) if generation_legends else 1.0
        self.metrics.recall = n_correct / len(gt_legends) if gt_legends else 1.0
        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1 = 2 * self.metrics.precision * self.metrics.recall / (self.metrics.precision + self.metrics.recall)
        else:
            self.metrics.f1 = 0.0

# --- Main Flow & Parallel Processing ---
def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path, use_position: bool) -> Tuple[str, LegendMetrics]:
    """Processes a single pair of generated and ground-truth code files."""
    logger.info(f"[{file_name}] Starting processing...")
    evaluator = LegendEvaluator(use_position=use_position)
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

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None, use_position: bool = True) -> Dict[str, LegendMetrics]:
    """Evaluates all matching Python files in two directories in parallel."""
    generation_path = Path(generation_dir)
    gt_path = Path(gt_dir)
    
    common_files = sorted(list(set(f.name for f in generation_path.glob("*.py")) & set(f.name for f in gt_path.glob("*.py"))))
    if not common_files:
        logger.warning("No matching file pairs found"); return {}
        
    if num_workers is None: num_workers = os.cpu_count()
    logger.info(f"Found {len(common_files)} file pairs to process using {num_workers} workers.")
    
    all_results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_single_file, fname, generation_path, gt_path, use_position): fname for fname in common_files}
        
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                _, metrics = future.result()
                all_results[file_name] = metrics
            except Exception as e:
                logger.error(f"A critical error occurred while processing the future for {file_name}: {e}")
                all_results[file_name] = LegendMetrics(status=ExecutionStatus.FAILED, error_message=str(e))
                
    if output_file:
        save_results_to_json(all_results, output_file, generation_dir, gt_dir)
        
    return all_results

def save_results_to_json(results: Dict[str, LegendMetrics], output_file: str, generation_dir: str, gt_dir: str) -> None:
    """Saves the evaluation results to a JSON file with metadata."""
    json_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(), 
            "evaluator": "LegendEvaluator"
        }, 
        "individual_results": []
    }
    
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
    print("Legend Evaluation Script")
    
    generation_dir = PROJECT_PATH / "gpt4o_code"
    gt_dir = PROJECT_PATH / "level1_test"
    
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure '{generation_dir.name}' and '{gt_dir.name}' directories exist at: {PROJECT_PATH}")
        return
        
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "legend_evaluation_results.json"),
            num_workers=os.cpu_count(),
            use_position=True
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


