# parameter_evaluation.py
from typing import List, Tuple, Any, Dict, Optional
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import logging
import json
from datetime import datetime
import runpy
from enum import Enum
import io
from contextlib import redirect_stdout
from collections import Counter
import numpy as np

# --- Core Dependencies & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
PROJECT_PATH = Path(os.environ.get("PROJECT_PATH", Path(__file__).parent.resolve()))
sys.path.insert(0, str(PROJECT_PATH))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# --- Status & Data Classes ---
class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ScoreBlock:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

@dataclass
class ParameterMetrics:
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: str = ""
    data_metrics: ScoreBlock = field(default_factory=ScoreBlock)
    visual_metrics: ScoreBlock = field(default_factory=ScoreBlock)
    
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

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# --- Parameter Evaluator Class ---
class ParameterEvaluator:
    def __init__(self) -> None:
        self.metrics = ParameterMetrics()
        self.DATA_PARAM_KEYS = {'xdata', 'ydata', 'offsets', 'xy', 'verts', 'width', 'height', 'sizes'}
        self.IGNORED_PARAMS = {'color', 'c', 'colors', 'label', 'labels', 'edgecolor', 'facecolor'}

    def __call__(self, gen_fig: Optional[plt.Figure], gt_fig: Optional[plt.Figure]) -> ParameterMetrics:
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not get a valid Figure object"
            return self.metrics
        try:
            gen_params = self._extract_params_from_figure(gen_fig)
            gt_params = self._extract_params_from_figure(gt_fig)
            self._calculate_strict_metrics(gen_params, gt_params)
        except Exception as e:
            logger.error(f"Error during parameter evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_params_from_figure(self, fig: plt.Figure) -> List[Dict]:
        """Extracts plotting parameters by introspecting artists from a Figure object."""
        extracted_params = []
        for ax in fig.axes:
            # Extract from lines (plot, etc.)
            for line in ax.lines:
                params = {
                    'type': 'line', 'xdata': np.array(line.get_xdata()).tolist(), 'ydata': np.array(line.get_ydata()).tolist(),
                    'linestyle': line.get_linestyle(), 'linewidth': line.get_linewidth(), 'marker': line.get_marker(),
                    'markersize': line.get_markersize(), 'alpha': line.get_alpha()
                }
                extracted_params.append(params)
            
            # Differentiate between different types of patches
            for patch in ax.patches:
                params = {'alpha': patch.get_alpha()}
                # For Rectangles (from bar, hist)
                if isinstance(patch, Rectangle):
                    params.update({
                        'type': 'rectangle_patch',
                        'xy': np.array(patch.get_xy()).tolist(),
                        'width': patch.get_width(),
                        'height': patch.get_height(),
                    })
                    extracted_params.append(params)
                # For Polygons (from fill, violinplot)
                elif isinstance(patch, Polygon):
                    params.update({
                        'type': 'polygon_patch',
                        'verts': np.array(patch.get_xy()).tolist(),
                    })
                    extracted_params.append(params)
                # Other patch types (e.g., Circle, Ellipse) could be added here
            
            # Extract from collections (scatter, etc.)
            for collection in ax.collections:
                params = {'type': 'collection', 'alpha': collection.get_alpha()}
                if hasattr(collection, 'get_offsets'):
                    params['offsets'] = np.array(collection.get_offsets()).tolist()
                if hasattr(collection, 'get_sizes'):
                    params['sizes'] = np.array(collection.get_sizes()).tolist()
                # Only add if data-related parameters were found
                if len(params) > 2: 
                    extracted_params.append(params)
        return extracted_params

    def _calculate_value_similarity(self, val1: Any, val2: Any) -> float:
        """Strictly compares two values, handling numerics, strings, and lists/arrays."""
        if val1 is None and val2 is None: return 1.0
        if val1 is None or val2 is None: return 0.0
        
        try:
            if isinstance(val1, str): val1 = float(val1)
            if isinstance(val2, str): val2 = float(val2)
        except (ValueError, TypeError):
            pass

        if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
            return 1.0 if np.isclose(val1, val2) else 0.0
        if isinstance(val1, (bool, str)):
            return 1.0 if str(val1) == str(val2) else 0.0
        if isinstance(val1, (list, np.ndarray)):
            if not isinstance(val2, (list, np.ndarray)): return 0.0
            if not len(val1) and not len(val2): return 1.0
            if not len(val1) or not len(val2): return 0.0
            try: # Attempt numeric Jaccard similarity
                v1 = np.asarray(val1, dtype=float).flatten()
                v2 = np.asarray(val2, dtype=float).flatten()
                intersection = np.intersect1d(v1, v2).size
                union = np.union1d(v1, v2).size
                return intersection / union if union > 0 else 1.0
            except (ValueError, TypeError): # Fallback to string-based set comparison
                set1, set2 = set(str(v) for v in val1), set(str(v) for v in val2)
                return len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 1.0
        return 0.0

    def _calculate_strict_metrics(self, gen_elements: List[Dict], gt_elements: List[Dict]):
        """Calculates precision, recall, and F1 for data and visual parameters."""
        if not gen_elements and not gt_elements:
            self.metrics.data_metrics = self.metrics.visual_metrics = ScoreBlock(1.0, 1.0, 1.0)
            return

        total_data_score, total_visual_score = 0.0, 0.0
        gt_data_count, gt_visual_count = 0, 0
        gen_data_count, gen_visual_count = 0, 0

        unmatched_gen_elements = gen_elements[:]
        for gt_elem in gt_elements:
            best_score, best_match_index = -1.0, -1
            # Find the best matching generated element based on overall parameter similarity
            for i, gen_elem in enumerate(unmatched_gen_elements):
                if gt_elem.get('type') == gen_elem.get('type'):
                    current_score = sum(self._calculate_value_similarity(gt_elem.get(k), gen_elem.get(k)) for k in gt_elem if k != 'type')
                    if current_score > best_score:
                        best_score, best_match_index = current_score, i
            
            if best_match_index != -1:
                matched_gen_elem = unmatched_gen_elements.pop(best_match_index)
                all_keys = set(gt_elem.keys()) | set(matched_gen_elem.keys())
                # Calculate scores for matched elements
                for key in all_keys:
                    if key in self.IGNORED_PARAMS or key == 'type': continue
                    category = 'data' if key in self.DATA_PARAM_KEYS else 'visual'
                    gt_val, gen_val = gt_elem.get(key), matched_gen_elem.get(key)
                    score = self._calculate_value_similarity(gt_val, gen_val)
                    if category == 'data': total_data_score += score
                    else: total_visual_score += score
        
        # Count total parameters for ground truth
        for gt_elem in gt_elements:
            for key in gt_elem:
                if key in self.IGNORED_PARAMS or key == 'type': continue
                if key in self.DATA_PARAM_KEYS: gt_data_count += 1
                else: gt_visual_count += 1
        
        # Count total parameters for generated code
        for gen_elem in gen_elements:
             for key in gen_elem:
                if key in self.IGNORED_PARAMS or key == 'type': continue
                if key in self.DATA_PARAM_KEYS: gen_data_count += 1
                else: gen_visual_count += 1

        # Calculate data metrics
        data_p = total_data_score / gen_data_count if gen_data_count > 0 else 1.0 if not gt_data_count else 0.0
        data_r = total_data_score / gt_data_count if gt_data_count > 0 else 1.0 if not gen_data_count else 0.0
        data_f1 = 2 * (data_p * data_r) / (data_p + data_r) if (data_p + data_r) > 0 else 0.0
        self.metrics.data_metrics = ScoreBlock(data_p, data_r, data_f1)
        
        # Calculate visual metrics
        visual_p = total_visual_score / gen_visual_count if gen_visual_count > 0 else 1.0 if not gt_visual_count else 0.0
        visual_r = total_visual_score / gt_visual_count if gt_visual_count > 0 else 1.0 if not gen_visual_count else 0.0
        visual_f1 = 2 * (visual_p * visual_r) / (visual_p + visual_r) if (visual_p + visual_r) > 0 else 0.0
        self.metrics.visual_metrics = ScoreBlock(visual_p, visual_r, visual_f1)

# --- Main Flow & Parallel Processing ---
def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path) -> Tuple[str, ParameterMetrics]:
    """Processes a single pair of generated and ground-truth code files."""
    logger.info(f"[{file_name}] Starting processing...")
    evaluator = ParameterEvaluator()
    gen_fig, gen_err = execute_code_and_get_figure(str(generation_dir / file_name))
    gt_fig, gt_err = execute_code_and_get_figure(str(gt_dir / file_name))
    
    metrics = evaluator(gen_fig, gt_fig)
    
    if gen_err or gt_err:
        if metrics.status == ExecutionStatus.SUCCESS: metrics.status = ExecutionStatus.FAILED
        if "timed out" in str(gen_err) or "timed out" in str(gt_err): metrics.status = ExecutionStatus.TIMEOUT
        metrics.error_message = f"GenErr: {gen_err}; GtErr: {gt_err}"
        logger.warning(f"[{file_name}] Processing failed: {metrics.error_message}")
    else:
        logger.info(f"[{file_name}] Processing successful (Data F1:{metrics.data_metrics.f1:.2f} Visual F1:{metrics.visual_metrics.f1:.2f})")
        
    return file_name, metrics

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None) -> Dict[str, ParameterMetrics]:
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
        future_to_file = {executor.submit(process_single_file, fname, generation_path, gt_path): fname for fname in common_files}
        
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                _, metrics = future.result()
                all_results[file_name] = metrics
            except Exception as e:
                logger.error(f"A critical error occurred while processing the future for {file_name}: {e}")
                all_results[file_name] = ParameterMetrics(status=ExecutionStatus.FAILED, error_message=str(e))
                
    if output_file:
        save_results_to_json(all_results, output_file, generation_dir, gt_dir)
        
    return all_results

def save_results_to_json(results: Dict[str, ParameterMetrics], output_file: str, generation_dir: str, gt_dir: str) -> None:
    """Saves the evaluation results to a JSON file with metadata."""
    json_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(), 
            "evaluator": "ParameterEvaluator"
        }, 
        "individual_results": []
    }
    
    for file_name, metrics in sorted(results.items()):
        json_data["individual_results"].append({
            "file": file_name,
            "status": metrics.status.value,
            "error_message": metrics.error_message,
            "data_metrics": {k:round(v,4) for k,v in metrics.data_metrics.__dict__.items()},
            "visual_metrics": {k:round(v,4) for k,v in metrics.visual_metrics.__dict__.items()}
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
    logger.info(f"Evaluation results saved to: {output_file}")

def main():
    print("=" * 60)
    print("Parameter Evaluation Script")
    
    generation_dir = PROJECT_PATH / "generation_code"
    gt_dir = PROJECT_PATH / "gt_code"
    
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure '{generation_dir.name}' and '{gt_dir.name}' directories exist at: {PROJECT_PATH}")
        return
        
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "parameter_evaluation_results.json"),
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


