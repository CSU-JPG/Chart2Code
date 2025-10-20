# -*- coding: utf-8 -*-
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
from itertools import permutations
import numpy as np
import io
from contextlib import redirect_stdout
from collections import Counter, defaultdict

# --- Core Dependencies & Configuration ---
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
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Functions ---
# To make the script self-contained, helper functions are defined here.

def calculate_color_similarity(color1_hex: str, color2_hex: str) -> float:
    """Calculates similarity between two hex colors based on Euclidean distance in RGB space."""
    if color1_hex == color2_hex:
        return 1.0
    try:
        c1 = tuple(int(color1_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        c2 = tuple(int(color2_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        dist = np.sqrt(sum([(a - b)**2 for a, b in zip(c1, c2)]))
        max_dist = np.sqrt(3 * (255**2)) # Max possible distance in RGB cube.
        return 1.0 - (dist / max_dist)
    except (ValueError, TypeError):
        return 0.0

def convert_color_to_hex(color: Any) -> Optional[str]:
    """Converts various matplotlib color formats to a hex string, ignoring transparent colors."""
    from matplotlib.colors import to_hex, to_rgba
    try:
        # Ignore None or 'none' string literal
        if color is None or (isinstance(color, str) and color.lower() == 'none'):
            return None
        rgba = to_rgba(color)
        # Ignore fully transparent colors
        if rgba[3] == 0:
            return None
        return to_hex(rgba, keep_alpha=False).upper()
    except (ValueError, TypeError):
        return None

# --- Status & Data Classes ---

class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class ColorMetrics:
    """Stores the evaluation results for color comparison."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_similarity: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: str = ""

# --- Code Execution ---

def _execute_code_runner(code_file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Executes a script in a sandboxed manner, suppressing output and plots.
    This function is designed to be run in a separate process.
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Suppress plot rendering and close all figures to save memory
    plt.show = lambda *args, **kwargs: None
    plt.savefig = lambda *args, **kwargs: None
    plt.close('all')
    
    output_buffer = io.StringIO()
    try:
        # Redirect stdout to prevent printing to the console
        with redirect_stdout(output_buffer):
            runpy.run_path(str(code_file_path), run_name='__main__')
        return True, None
    except Exception as e:
        return False, f"Code execution failed: {e}"
    finally:
        plt.close('all')

def execute_code_and_get_figure(code_file_path: str, timeout: int = 60) -> Tuple[Optional[Figure], Optional[str]]:
    """
    Executes a plotting script and retrieves the last generated matplotlib figure.
    Uses a ProcessPoolExecutor to enforce a timeout and isolate execution.
    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_code_runner, code_file_path)
        try:
            success, error_msg = future.result(timeout=timeout)
            if not success:
                return None, error_msg
        except TimeoutError:
            return None, f"Execution timed out after {timeout} seconds"
        except Exception as e:
            return None, f"Executor encountered an unknown error: {e}"


    plt.close('all')
    try:
        runpy.run_path(str(code_file_path), run_name='__main__')
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return None, "Script executed successfully but produced no figure"
        # Return the last created figure
        return plt.figure(fig_nums[-1]), None
    except Exception as e:
        return None, f"Error while retrieving figure object: {e}"
    finally:
        plt.close('all')

# ---  Data-Aware Expert Evaluator ---

class ColorEvaluator:
    """
    A sophisticated evaluator that extracts and compares colors from plots
    by associating them with data elements (e.g., labels, categories).
    """

    TYPE_WEIGHTS = {
        # --- Key Data Elements (High Importance) ---
        'patch_face': 1.0,      # Bar color, pie slice color
        'line_color': 1.0,      # Line plot color
        'scatter_color': 1.0,   # Scatter plot color (for distinct groups)
        'text_color': 1.0,      # WordCloud text color
        
        # --- Palettes / Less Distinct Elements (Medium Importance) ---
        'scatter_palette': 0.7, # Scatter plot palette (color as a variable)
        'poly3d_palette': 0.7,  # 3D surface plot palette
        
        # --- Aesthetic/Chrome Elements (Low Importance) ---
        'patch_edge': 0.01,
        'axes_bg': 0.01,
        'figure_bg': 0.01,
        'spine': 0.01,
        'tick_label': 0.05,
        'axis_label': 0.05,
        'title': 0.05,
        'legend_text': 0.05,
        'legend_bg': 0.01,
    }
    DEFAULT_WEIGHT = 0.1

    def __init__(self) -> None:
        self.metrics = ColorMetrics()

    def __call__(self, gen_fig: Optional[Figure], gt_fig: Optional[Figure]) -> ColorMetrics:
        """Evaluates the colors of a generated figure against a ground truth figure."""
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not retrieve a valid Figure object from generated or GT code."
            return self.metrics
        try:
            generation_data = self._extract_colors_from_figure_expert(gen_fig)
            gt_data = self._extract_colors_from_figure_expert(gt_fig)
            self._calculate_metrics(generation_data, gt_data)
        except Exception as e:
            logger.error(f"Error during color evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_colors_from_figure_expert(self, figure: Figure) -> Dict[str, Dict[str, str]]:
        """
        Extracts a data-aware mapping of {element_type: {data_key: color}}.
        This method inspects fundamental matplotlib artists to support a wide variety of plots.
        """
        extracted_data = defaultdict(dict)
        fallback_counters = defaultdict(int) # Used when a data key (e.g., label) cannot be found.

        # Extract global figure properties
        if color := convert_color_to_hex(figure.patch.get_facecolor()):
            extracted_data['figure_bg']['figure'] = color

        for ax in figure.axes:
            if color := convert_color_to_hex(ax.patch.get_facecolor()):
                extracted_data['axes_bg'][f'ax_{id(ax)}'] = color
            
            # Strategy 1: Use the legend for high-quality data-color mapping.
            if ax.get_legend():
                for handle, label in zip(ax.get_legend().legend_handles, ax.get_legend().get_texts()):
                    key = label.get_text()
                    color = None
                    if hasattr(handle, 'get_facecolor'):
                        color = convert_color_to_hex(handle.get_facecolor())
                    elif hasattr(handle, 'get_color'):
                        color = convert_color_to_hex(handle.get_color())
                    
                    if color and key:
                        # Differentiate between patch-based legends (bars) and line-based legends
                        if isinstance(handle, plt.Rectangle):
                            extracted_data['patch_face'][key] = color
                        else:
                            extracted_data['line_color'][key] = color

            # Strategy 2: Extract from Patches (for bar, pie, etc.).
            try:
                tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
                for i, patch in enumerate(ax.patches):
                    if color := convert_color_to_hex(patch.get_facecolor()):
                        # Try to link to a tick label; otherwise, use a fallback key.
                        key = tick_labels[i] if i < len(tick_labels) and tick_labels[i] else None
                        if not key:
                            key = f"patch_{fallback_counters['patch_face']}"
                            fallback_counters['patch_face'] += 1
                        if key not in extracted_data['patch_face']:
                            extracted_data['patch_face'][key] = color
                    
                    if e_color := convert_color_to_hex(patch.get_edgecolor()):
                        key = tick_labels[i] if i < len(tick_labels) and tick_labels[i] else f"patch_edge_{i}"
                        extracted_data['patch_edge'][key] = e_color
            except Exception as e:
                logger.warning(f"Error processing Patches: {e}")

            # Strategy 3: Extract from Lines (for line plots, etc.).
            try:
                for line in ax.lines:
                    if color := convert_color_to_hex(line.get_color()):
                        key = line.get_label()
                        # Use a fallback key for unlabeled lines (labels starting with '_' are internal).
                        if not key or key.startswith('_'):
                            key = f"line_{fallback_counters['line_color']}"
                            fallback_counters['line_color'] += 1
                        if key not in extracted_data['line_color']:
                            extracted_data['line_color'][key] = color
            except Exception as e:
                logger.warning(f"Error processing Lines: {e}")

            # Strategy 4: Extract from Collections (for scatter, heatmap, etc.).
            try:
                for collection in ax.collections:
                    colors = collection.get_facecolors()
                    if len(colors) == 0: continue
                    
                    # Case A: All points in the collection have the same color (a single group).
                    if len(set(map(tuple, colors))) == 1:
                        if color := convert_color_to_hex(colors[0]):
                            key = collection.get_label()
                            if not key or key.startswith('_'):
                                key = f"scatter_group_{fallback_counters['scatter_color']}"
                                fallback_counters['scatter_color'] += 1
                            if key not in extracted_data['scatter_color']:
                                extracted_data['scatter_color'][key] = color
                    # Case B: Multiple colors found, treat as a palette.
                    else:
                        unique_colors = {convert_color_to_hex(c) for c in colors if c is not None}
                        for i, color in enumerate(unique_colors):
                            key = f"palette_color_{fallback_counters['scatter_palette']}"
                            fallback_counters['scatter_palette'] += 1
                            extracted_data['scatter_palette'][key] = color
            except Exception as e:
                logger.warning(f"Error processing Collections: {e}")

            # Strategy 5: Extract from Texts (for WordCloud, annotations).
            try:
                for text in ax.texts:
                    if color := convert_color_to_hex(text.get_color()):
                        key = text.get_text()
                        if key: extracted_data['text_color'][key] = color
            except Exception as e:
                logger.warning(f"Error processing Texts: {e}")
            
            # Strategy 6: Extract from static text elements.
            if (color := convert_color_to_hex(ax.title.get_color())): extracted_data['title']['title'] = color
            if (color := convert_color_to_hex(ax.xaxis.label.get_color())): extracted_data['axis_label']['xlabel'] = color
            if (color := convert_color_to_hex(ax.yaxis.label.get_color())): extracted_data['axis_label']['ylabel'] = color

        return dict(extracted_data)
        
    def _calculate_metrics(self, generation_data: Dict[str, Dict[str, str]], gt_data: Dict[str, Dict[str, str]]) -> None:
        """
        Calculates precision, recall, and F1 score based on a structure-aware comparison.
        It only compares elements of the same type (e.g., 'patch_face' to 'patch_face').
        Mismatched element types contribute to false positives and false negatives.
        """
        total_weighted_similarity = 0.0  # Numerator (True Positives)
        total_gen_weight = 0.0           # Denominator for Precision
        total_gt_weight = 0.0            # Denominator for Recall

        # 1. Calculate the total possible weighted score for each figure.
        for element_type, data_map in generation_data.items():
            weight = self.TYPE_WEIGHTS.get(element_type, self.DEFAULT_WEIGHT)
            total_gen_weight += len(data_map) * weight
        
        for element_type, data_map in gt_data.items():
            weight = self.TYPE_WEIGHTS.get(element_type, self.DEFAULT_WEIGHT)
            total_gt_weight += len(data_map) * weight

        # If both figures are empty, it's a perfect match.
        if total_gen_weight == 0 and total_gt_weight == 0:
            self.metrics.precision, self.metrics.recall, self.metrics.f1, self.metrics.total_similarity = 1.0, 1.0, 1.0, 1.0
            return

        # 2. Calculate weighted similarity only for common element types and data keys.
        common_element_types = set(generation_data.keys()) & set(gt_data.keys())

        for element_type in common_element_types:
            gen_map = generation_data[element_type]
            gt_map = gt_data[element_type]
            weight = self.TYPE_WEIGHTS.get(element_type, self.DEFAULT_WEIGHT)
            
            # Find common data keys (e.g., labels) within this element type.
            common_keys = set(gen_map.keys()) & set(gt_map.keys())
            
            for key in common_keys:
                gen_color = gen_map[key]
                gt_color = gt_map[key]
                
                # Similarity is only calculated for elements that match in both type and key.
                similarity = calculate_color_similarity(gen_color, gt_color)
                total_weighted_similarity += similarity * weight

        # 3. Finalize and update metric scores.
        self._update_weighted_metrics(total_weighted_similarity, total_gen_weight, total_gt_weight)


    def _update_weighted_metrics(self, total_weighted_similarity: float, total_gen_weight: float, total_gt_weight: float) -> None:
        """Calculates and stores the final Precision, Recall, and F1 scores."""
        self.metrics.precision = total_weighted_similarity / total_gen_weight if total_gen_weight > 0 else 0.0
        self.metrics.recall = total_weighted_similarity / total_gt_weight if total_gt_weight > 0 else 0.0
        
        # Clamp values between 0.0 and 1.0 to handle potential floating point inaccuracies.
        self.metrics.precision = max(0.0, min(1.0, self.metrics.precision))
        self.metrics.recall = max(0.0, min(1.0, self.metrics.recall))
        
        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1 = 2 * self.metrics.precision * self.metrics.recall / (self.metrics.precision + self.metrics.recall)
        else: 
            self.metrics.f1 = 0.0
        self.metrics.total_similarity = total_weighted_similarity

# --- Main Workflow & Parallel Processing ---

def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path) -> Tuple[str, ColorMetrics]:
    """
    Processes a single pair of generated and ground truth scripts.
    """
    logger.info(f"Processing: {file_name}...")
    evaluator = ColorEvaluator()
    
    gen_fig, gen_err = execute_code_and_get_figure(str(generation_dir / file_name))
    gt_fig, gt_err = execute_code_and_get_figure(str(gt_dir / file_name))
    
    metrics = evaluator(gen_fig, gt_fig)
    
    # Consolidate execution errors with evaluation metrics.
    if gen_err or gt_err:
        if metrics.status == ExecutionStatus.SUCCESS:
            metrics.status = ExecutionStatus.FAILED
        if "timeout" in str(gen_err).lower() or "timeout" in str(gt_err).lower():
            metrics.status = ExecutionStatus.TIMEOUT
        metrics.error_message = f"GenError: {gen_err}; GtError: {gt_err}"
        logger.warning(f"Failed to process {file_name}: {metrics.error_message}")
    else:
        logger.info(f"Finished {file_name} (P:{metrics.precision:.2f} R:{metrics.recall:.2f} F1:{metrics.f1:.2f})")
        
    return file_name, metrics

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None) -> Dict[str, ColorMetrics]:
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
        
    logger.info(f"Found {len(common_files)} files to process using {num_workers} workers.")
    
    all_results = {}
    
    # Use ProcessPoolExecutor to run evaluations in parallel.
    with ProcessPoolExecutor(max_workers=num_workers, max_tasks_per_child=20) as executor:
        future_to_file = {executor.submit(process_single_file, fname, generation_path, gt_path): fname for fname in common_files}
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                _, metrics = future.result()
                all_results[file_name] = metrics
            except Exception as e:
                logger.error(f"A critical error occurred while processing future for {file_name}: {e}")
                all_results[file_name] = ColorMetrics(status=ExecutionStatus.FAILED, error_message=str(e))

    if output_file:
        save_results_to_json(all_results, output_file, generation_dir, gt_dir)
        
    return all_results

def save_results_to_json(results: Dict[str, ColorMetrics], output_file: str, generation_dir: str, gt_dir: str) -> None:
    """Saves the aggregated evaluation results to a JSON file."""
    json_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "evaluator": "ColorEvaluator"
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
            "total_similarity": round(metrics.total_similarity, 4),
            "error_message": metrics.error_message
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Evaluation results saved to: {output_file}")

def main():
    """Main entry point for the script."""
    print("=" * 60)
    print("Color Evaluator")
    
    generation_dir = PROJECT_PATH / "generation_code"
    gt_dir = PROJECT_PATH / "gt_code"
    
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure 'generation_code' and 'gt_code' directories exist in: {PROJECT_PATH}")
        return
        
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "color_evaluation_results_data_aware.json"),
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

