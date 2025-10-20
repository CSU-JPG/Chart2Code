# text_evaluator.py
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

# For stricter text comparison, install the library: pip install python-Levenshtein
try:
    from Levenshtein import ratio as levenshtein_ratio
except ImportError:
    logger.warning("python-Levenshtein not found. Using basic string comparison. For stricter evaluation, run 'pip install python-Levenshtein'")
    def levenshtein_ratio(s1, s2):
        return 1.0 if s1 == s2 else 0.0

# --- Status & Data Classes ---
class ExecutionStatus(Enum):
    """Enumeration for the execution status of a script."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class TextMetrics:
    """Dataclass to hold text evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: str = ""

# --- Timeout-Protected & Output-Suppressed Code Executor ---
def _execute_code_runner(code_file_path: str) -> Tuple[bool, Optional[str]]:
    """Runs code in an isolated process to check for errors, suppressing all stdout."""
    import runpy, matplotlib, io
    from contextlib import redirect_stdout
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Disable interactive plotting and close all figures to ensure a clean state.
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
    """Executes plotting code with a timeout and returns the generated Figure object."""
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


# --- Hardened Evaluator Class with Strict Logic ---
class TextEvaluator:
    """Evaluates text similarity between two matplotlib Figure objects."""
    def __init__(self) -> None:
        self.metrics = TextMetrics()

    def __call__(self, gen_fig: Optional[plt.Figure], gt_fig: Optional[plt.Figure]) -> TextMetrics:
        """Evaluates the text of a generated figure against a ground truth figure."""
        if gen_fig is None or gt_fig is None:
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = "Could not get a valid Figure object for comparison."
            return self.metrics
        try:
            generation_texts = self._extract_texts_from_figure(gen_fig)
            gt_texts = self._extract_texts_from_figure(gt_fig)
            self._calculate_metrics(generation_texts, gt_texts)
        except Exception as e:
            logger.error(f"Error during text evaluation: {e}", exc_info=True)
            self.metrics.status = ExecutionStatus.FAILED
            self.metrics.error_message = str(e)
        return self.metrics

    def _extract_texts_from_figure(self, fig: plt.Figure) -> Dict[str, List[str]]:
        """Extracts and categorizes all text elements from a matplotlib Figure."""
        texts = {
            "title": [], "xlabel": [], "ylabel": [], "tick_label": [],
            "suptitle": [], "legend_text": [], "annotation": []
        }
        if fig._suptitle and fig._suptitle.get_text():
            texts["suptitle"].append(fig._suptitle.get_text())

        for ax in fig.axes:
            if ax.title.get_text():
                texts["title"].append(ax.title.get_text())
            if ax.xaxis.label.get_text():
                texts["xlabel"].append(ax.xaxis.label.get_text())
            if ax.yaxis.label.get_text():
                texts["ylabel"].append(ax.yaxis.label.get_text())
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                if label.get_text():
                    texts["tick_label"].append(label.get_text())
            
            if legend := ax.get_legend():
                for text in legend.get_texts():
                    if text.get_text():
                        texts["legend_text"].append(text.get_text())
            
            # For text generated by ax.text() or ax.annotate()
            for text in ax.texts:
                if text.get_text():
                    texts["annotation"].append(text.get_text())
        
        # Remove categories with no text elements.
        return {k: v for k, v in texts.items() if v}

    def _calculate_metrics(self, generation_texts: Dict[str, List[str]], gt_texts: Dict[str, List[str]]) -> None:
        """Calculates precision, recall, and F1 based on categorized text similarity."""
        if not generation_texts and not gt_texts:
            self.metrics.precision = 1.0
            self.metrics.recall = 1.0
            self.metrics.f1 = 1.0
            return

        total_similarity_score = 0.0
        total_gt_text_count = sum(len(texts) for texts in gt_texts.values())
        total_gen_text_count = sum(len(texts) for texts in generation_texts.values())

        all_categories = set(gt_texts.keys()) | set(generation_texts.keys())

        for category in all_categories:
            gt_list = gt_texts.get(category, [])
            gen_list = generation_texts.get(category, [])
            
            if not gt_list or not gen_list:
                continue

            # Match generated texts to ground truth texts within the same category.
            unmatched_gt = gt_list[:]
            for gen_text in gen_list:
                if not unmatched_gt: break
                best_score = -1
                best_match_index = -1
                for i, gt_text in enumerate(unmatched_gt):
                    score = levenshtein_ratio(gen_text, gt_text)
                    if score > best_score:
                        best_score = score
                        best_match_index = i
                
                if best_match_index != -1:
                    total_similarity_score += best_score
                    unmatched_gt.pop(best_match_index)

        self.metrics.precision = total_similarity_score / total_gen_text_count if total_gen_text_count > 0 else 1.0 if not gt_texts else 0.0
        self.metrics.recall = total_similarity_score / total_gt_text_count if total_gt_text_count > 0 else 1.0 if not generation_texts else 0.0
        
        if self.metrics.precision + self.metrics.recall > 0:
            self.metrics.f1 = 2 * self.metrics.precision * self.metrics.recall / (self.metrics.precision + self.metrics.recall)
        else:
            self.metrics.f1 = 0.0

# --- Main Flow & Parallel Processing ---
def process_single_file(file_name: str, generation_dir: Path, gt_dir: Path) -> Tuple[str, TextMetrics]:
    """Processes a single pair of generated and ground truth code files."""
    logger.info(f"[{file_name}] Starting processing...")
    evaluator = TextEvaluator()
    gen_fig, gen_err = execute_code_and_get_figure(str(generation_dir / file_name))
    gt_fig, gt_err = execute_code_and_get_figure(str(gt_dir / file_name))
    
    metrics = evaluator(gen_fig, gt_fig)
    
    # Combine errors from execution and evaluation for a complete report.
    if gen_err or gt_err:
        if metrics.status == ExecutionStatus.SUCCESS: metrics.status = ExecutionStatus.FAILED
        if "timed out" in str(gen_err) or "timed out" in str(gt_err): metrics.status = ExecutionStatus.TIMEOUT
        metrics.error_message = f"GenErr: {gen_err}; GtErr: {gt_err}"
        logger.warning(f"[{file_name}] Processing failed: {metrics.error_message}")
    else:
        logger.info(f"[{file_name}] Processing successful (P:{metrics.precision:.2f} R:{metrics.recall:.2f} F1:{metrics.f1:.2f})")
    return file_name, metrics

def batch_evaluate_directory(generation_dir: str, gt_dir: str, output_file: Optional[str] = None, num_workers: Optional[int] = None) -> Dict[str, TextMetrics]:
    """Evaluates all matching Python files in two directories in parallel."""
    generation_path = Path(generation_dir)
    gt_path = Path(gt_dir)
    common_files = sorted(list(set(f.name for f in generation_path.glob("*.py")) & set(f.name for f in gt_path.glob("*.py"))))
    
    if not common_files:
        logger.warning("No matching file pairs found between the two directories.")
        return {}
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
                all_results[file_name] = TextMetrics(status=ExecutionStatus.FAILED, error_message=str(e))
                
    if output_file:
        save_results_to_json(all_results, output_file)
        
    return all_results

def save_results_to_json(results: Dict[str, TextMetrics], output_file: str) -> None:
    """Saves the evaluation results to a JSON file."""
    json_data = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(), 
            "evaluator": "TextEvaluator"
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
    print("Text-Based Plot Evaluation")
    print("=" * 60)
    
    # Define directories for generated code and ground truth code.
    generation_dir = PROJECT_PATH / "generation_code"
    gt_dir = PROJECT_PATH / "gt_code"
    
    if not generation_dir.exists() or not gt_dir.exists():
        logger.error(f"Error: Please ensure 'generation_code' and 'gt_code' directories exist at: {PROJECT_PATH}")
        return
        
    try:
        results = batch_evaluate_directory(
            generation_dir=str(generation_dir),
            gt_dir=str(gt_dir),
            output_file=str(PROJECT_PATH / "text_evaluation_results.json"),
            num_workers=os.cpu_count()
        )
        
        if results:
            print("\n" + "=" * 22 + " Evaluation Summary " + "=" * 22)
            status_counts = Counter(m.status.value for m in results.values())
            total = len(results)
            print(f"Total files evaluated: {total}")
            for status, count in status_counts.items():
                print(f"  - {status.capitalize():<10}: {count:4d} files ({count/total:.1%})")
            print("=" * 60)
            
    except Exception as e:
        logger.error(f"Batch evaluation failed due to a critical error: {e}", exc_info=True)

if __name__ == "__main__":
    main()


