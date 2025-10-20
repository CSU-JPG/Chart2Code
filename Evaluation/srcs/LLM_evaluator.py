import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import openai
from dotenv import load_dotenv

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("API_CodeEvaluator_Strict")

# --- Load environment variables ---
load_dotenv()

# --- Configure API client ---
try:
    client = openai.OpenAI(
        base_url=os.getenv('OPENAI_API_URL'),
        api_key=os.getenv('OPENAI_API_KEY'),
        timeout=float(os.getenv('OPENAI_TIMEOUT', 120.0)),
        max_retries=int(os.getenv('OPENAI_MAX_RETRIES', 3))
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    sys.exit(1)

class ExecutionStatus:
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

# --- System prompt for multi-step evaluation ---
STRICT_SYSTEM_PROMPT = """
You are a meticulous and strict expert Python data visualization analyst. Your task is to compare two Python plotting scripts and evaluate the visual similarity of their final outputs based on a SINGLE, specific dimension.

Your analysis must be based **solely on the provided code**. Do not execute it. Your evaluation must be critical and detail-oriented.

**Scoring Philosophy:** Assume a perfect score of 100, then **deduct points for every deviation** you find, no matter how minor. A score of 100 is reserved ONLY for scripts that produce visually indistinguishable plots.

You must return ONLY a single JSON object with two keys: "score" (an integer from 0 to 100) and "reason" (a concise, expert analysis in English). Do not include any other text in your response.
"""

# --- Core API call function ---
def _call_api_for_evaluation(
    model: str,
    gt_code: str,
    generation_code: str,
    prompt: str
) -> Dict[str, Any]:
    response_content = ""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
                        **Ground Truth Code:**
                        ```python
                        {gt_code}
                        ```

                        **Generated Code:**
                        ```python
                        {generation_code}
                        ```

                        **Evaluation Dimension & Rules:**
                        {prompt}
                        """
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        logger.error(f"Model {model} returned invalid JSON: {e}\nRaw Response: {response_content}", exc_info=False)
        return {"score": 0, "reason": f"API returned invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"Error calling model {model} for evaluation: {e}", exc_info=False)
        return {"score": 0, "reason": f"API call failed: {e}"}

# --- Worker function for processing a single file pair ---
def evaluate_and_save_single_file_api(
    gen_file_path: Path,
    gt_file_path: Path,
    evaluator_configs: Dict[str, Dict],
    weights: Dict[str, float],
    results_dir: Path,
    model_name: str
) -> Optional[Dict[str, Any]]:
    file_name = gen_file_path.name
    
    def read_code_file(path: Path) -> str:
        try: return path.read_text(encoding='utf-8')
        except UnicodeDecodeError: return path.read_text(encoding='gbk')

    try:
        gt_code_content = read_code_file(gt_file_path)
        gen_code_content = read_code_file(gen_file_path)
    except Exception as e:
        logger.warning(f"Failed to read file pair for {file_name}, skipping: {e}")
        return None

    logger.info(f"Starting evaluation for: {file_name}")
    
    file_results = {}
    
    with ThreadPoolExecutor(max_workers=len(evaluator_configs)) as executor:
        future_to_dim = {
            executor.submit(_call_api_for_evaluation, model_name, gt_code_content, gen_code_content, config['prompt']): dim_name
            for dim_name, config in evaluator_configs.items()
        }
        for future in as_completed(future_to_dim):
            dim_name = future_to_dim[future]
            result = future.result() # result() will propagate exceptions from the thread
            score = result.get('score')
            reason = result.get('reason', 'N/A')
            standardized_score = float(score) / 100.0 if score is not None and isinstance(score, (int, float)) else 0.0
            
            file_results[dim_name] = {
                "status": ExecutionStatus.SUCCESS if score is not None else ExecutionStatus.FAILED,
                "score": standardized_score,
                "reason": reason
            }

    overall_score = 0.0
    total_weight_for_file = 0.0
    for dim_name, result in file_results.items():
        if result['status'] == ExecutionStatus.SUCCESS:
            weight = weights.get(dim_name, 0)
            overall_score += result['score'] * weight
            total_weight_for_file += weight
    
    if total_weight_for_file > 0:
        overall_score /= total_weight_for_file

    file_results['overall_score'] = overall_score
    
    output_path = results_dir / f"{gen_file_path.stem}_api_report.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write results to {output_path}: {e}")

    summary_data = {"file_name": file_name, "overall_score": overall_score}
    for dim_name, result in file_results.items():
        if isinstance(result, dict):
            summary_data[f"{dim_name}_score"] = result.get('score', 0.0)
            summary_data[f"{dim_name}_status"] = result.get('status')

    logger.info(f"Finished evaluation for: {file_name}")
    return summary_data


class APIBasedCodeEvaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.evaluator_configs = {
            'data_handling_and_transformation': { 'prompt': """
                Critically evaluate the DATA SOURCE and its TRANSFORMATION.
                - Focus on: How the numerical data passed to the plotting function is generated.
                - Check: Hardcoded lists/arrays, `pandas` or `numpy` array creation (e.g., `np.linspace`), data filtering (`df[...]`), mathematical operations (`np.sin(x)`, `df['a'] * 100`), and data aggregation.

                **Scoring Rubric (Start at 100, deduct points):**
                - **-0 points:** Data generation and transformations are functionally identical (e.g., `[1, 2, 3]` vs `np.array([1, 2, 3])`).
                - **-5 points:** Trivial differences in floating-point precision that are visually unnoticeable (e.g., `np.pi` vs `3.14159`).
                - **-25 points:** Different data filtering or selection that results in a subset or different ordering of the same underlying data.
                - **-50 points:** A different mathematical transformation is applied to the same base data (e.g., `np.sin(x)` vs `np.cos(x)`).
                - **-75 points:** The fundamental data sources are different (e.g., plotting `df['col_A']` vs `df['col_B']`).
                - **-100 points:** Data is completely unrelated in source, shape, and scale.
                """, 'weight': 0.20 },
            'chart_type_and_mapping': { 'prompt': """
                Critically evaluate the CORE CHART TYPE and DATA-TO-VISUALS MAPPING.
                - Focus on: The primary plotting function call (e.g., `plt.plot`, `ax.bar`, `sns.heatmap`).
                - Check: Which variables are mapped to which axes (e.g., `x=df['time']`, `y=df['value']`) and other visual properties (`size=`, `hue=`).

                **Scoring Rubric (Start at 100, deduct points):**
                - **-0 points:** The exact same plotting function is used with the same data-to-axis mappings.
                - **-15 points:** A visually similar plot type is used (e.g., `plt.plot()` vs `plt.scatter()`).
                - **-50 points:** A different plot type is used, but it's still plausible for the data (e.g., `plt.bar()` vs `plt.plot()` for time series). The core data variables on the axes are the same.
                - **-75 points:** Key data mappings are swapped or incorrect (e.g., x and y axes are flipped; `x='sales', y='time'` vs `x='time', y='sales'`).
                - **-100 points:** A fundamentally different and inappropriate chart type is used (e.g., `plt.pie()` vs `sns.lineplot()`).
                """, 'weight': 0.25 },
            'visual_aesthetics': { 'prompt': """
                Critically evaluate the VISUAL AESTHETICS like colors, markers, and line styles.
                - Focus on: Explicitly set styling arguments.
                - Check: `color`, `linestyle` (or `ls`), `linewidth` (or `lw`), `marker`, `markersize`, `alpha`, `cmap` (for heatmaps/scatter), `palette` (for seaborn).

                **Scoring Rubric (Start at 100, deduct points):**
                - **-0 points:** All explicit style arguments are identical.
                - **-10 points:** A minor style attribute is different (e.g., `linewidth=1.5` vs `linewidth=2.0`, or `marker='o'` vs `marker='x'`).
                - **-30 points:** The primary color is different (e.g., `color='blue'` vs `color='green'`). Or, one uses a default color while the other specifies one.
                - **-50 points:** Multiple style attributes are different (e.g., color and linestyle).
                - **-75 points:** The overall aesthetic is completely different (e.g., a solid blue line vs a transparent, dashed red line with markers).
                """, 'weight': 0.20 },
            'labels_titles_and_legend': { 'prompt': """
                Critically evaluate all TEXTUAL ELEMENTS: labels, titles, and legends.
                - Focus on: The content and presence of all text.
                - Check: `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()`, `fig.suptitle()`, and the `label` argument in plotting calls used by `ax.legend()`.

                **Scoring Rubric (Start at 100, deduct points):**
                - **-0 points:** All text elements are present and have identical content.
                - **-5 points:** Minor, non-substantive differences exist (e.g., "Sales Data" vs "Sales data", or a minor typo).
                - **-20 points:** A text element is present in both, but the content is substantively different (e.g., "Sales in 2023" vs "Profit in 2024").
                - **-40 points:** A key text element is missing in one script (e.g., one has a title, the other does not).
                - **-60 points:** Multiple key text elements are missing or incorrect.
                - **-100 points:** No text elements are present in one or both scripts.
                """, 'weight': 0.15 },
            'figure_layout_and_axes': { 'prompt': """
                Critically evaluate the FIGURE LAYOUT and AXES configuration.
                - Focus on: The overall canvas, subplot structure, and axis properties.
                - Check: `plt.figure(figsize=...)`, `plt.subplots()`, axis limits (`ax.set_xlim`, `ax.set_ylim`), axis scales (`ax.set_xscale`), and axis direction (`ax.invert_yaxis()`).

                **Scoring Rubric (Start at 100, deduct points):**
                - **-0 points:** Figure size, subplot structure, limits, and scales are all identical.
                - **-10 points:** Figure size is different, but the aspect ratio is similar.
                - **-25 points:** Axis limits are different, but the data range shown is largely the same.
                - **-50 points:** Axis scales are different (e.g., `linear` vs `log`). This is a major visual change.
                - **-75 points:** The subplot structure is different (e.g., `subplots(1, 2)` vs `subplots(2, 1)`).
                - **-100 points:** Completely different layouts (e.g., single plot vs. a complex grid of subplots).
                """, 'weight': 0.15 },
            'auxiliary_elements_and_ticks': { 'prompt': """
                Critically evaluate AUXILIARY elements, grid, spines, and ticks.
                - Focus on: Non-data visual elements that provide context or structure.
                - Check: `ax.grid()`, `ax.axhline()`, `ax.axvspan()`, `ax.spines[...]`, `ax.tick_params()`, and explicit tick setting (`ax.set_xticks`).

                **Scoring Rubric (Start at 100, deduct points):**
                - **-0 points:** All auxiliary elements and tick configurations are identical.
                - **-15 points:** An element is present in both but with different styling (e.g., a solid grid vs a dashed grid). Or, tick label formatting differs.
                - **-30 points:** An important element is present in one but missing in the other (e.g., one script calls `ax.grid(True)` and the other does not).
                - **-50 points:** A major contextual element is missing (e.g., a crucial `ax.axhline(y=0, ...)` that indicates a baseline). Or, spines are hidden in one but not the other.
                - **-75 points:** Major differences in tick locations (e.g., `xticks` are explicitly set to different values).
                """, 'weight': 0.05 }
        } # NOTE: Prompts truncated for brevity. Use your full prompts here.

        self.weights = {dim: config['weight'] for dim, config in self.evaluator_configs.items()}
        if weights is not None:
            self.weights.update(weights)
        total_weight = sum(self.weights.values())
        if not (0.999 < total_weight < 1.001):
            logger.warning(f"Weights do not sum to 1 (current sum: {total_weight:.4f}). Normalizing weights.")
            self.weights = {dim: w / total_weight for dim, w in self.weights.items()}

    def _generate_summary_report(self, summary_data_list: List[Dict], generation_dir: str, gt_json_path: str, output_path: str, model_name: str):
        if not summary_data_list:
            logger.warning("No data available to generate a summary report.")
            return

        df = pd.DataFrame(summary_data_list)
        
        avg_scores_by_dimension = {}
        for dim_name in self.weights.keys():
            score_col = f'{dim_name}_score'
            status_col = f'{dim_name}_status'
            if score_col in df.columns and status_col in df.columns:
                successful_scores = df[df[status_col] == ExecutionStatus.SUCCESS][score_col]
                avg_scores_by_dimension[dim_name] = successful_scores.astype(float).mean() if not successful_scores.empty else 0.0

        successful_evaluations = len(df[df['overall_score'] > 0])
        total_files_found = len(df)
        
        summary = {
            "total_files_found": total_files_found,
            "successful_evaluations": successful_evaluations,
            "average_overall_score": df['overall_score'].astype(float).mean() if 'overall_score' in df.columns and not df.empty else 0.0,
            "average_scores_by_dimension": avg_scores_by_dimension,
            "weights_used": self.weights
        }

        final_report = {
            "report_info": {
                "timestamp": datetime.now().isoformat(),
                "generation_directory": str(generation_dir),
                "gt_json_file": str(gt_json_path),
                "evaluator": "APIBasedCodeEvaluator_StrictMultiStep_V2",
                "model_used": model_name
            },
            "summary_statistics": summary,
            "individual_file_results": df.to_dict('records')
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report generated successfully: {output_path}")
        if summary:
            avg_score = summary.get('average_overall_score', 0.0)
            logger.info(f"Average overall score for model {model_name}: {avg_score:.4f}")
            if total_files_found > 0:
                success_rate = (successful_evaluations / total_files_found) * 100
                logger.info(f"Evaluation success rate: {successful_evaluations} / {total_files_found} ({success_rate:.2f}%)")
            else:
                logger.info("Evaluation success rate: 0 / 0 (0.00%)")

    def evaluate_and_report(
        self,
        generation_dir: str,
        gt_json_path: str,
        summary_json_path: str,
        details_dir: str,
        model_name: str,
        num_workers: Optional[int] = None
    ):
        gen_path = Path(generation_dir)
        gt_json = Path(gt_json_path)
        summary_output_path = Path(summary_json_path)
        results_dir = Path(details_dir)
        
        try:
            with open(gt_json, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not read or parse the Ground-Truth JSON file: {gt_json}. Error: {e}")
            return
        
        gt_base_path = gt_json.parent
        gt_map = { Path(item['GT code']).name: gt_base_path / item['GT code'] for item in gt_data if 'GT code' in item }
        gen_files = {f.name: f for f in gen_path.glob("*.py")}
        
        tasks: List[Tuple[Path, Path]] = []
        for gen_name, gen_file_path in gen_files.items():
            if gen_name in gt_map:
                tasks.append((gen_file_path, gt_map[gen_name]))

        if not tasks:
            logger.error(f"No common .py files found between {gen_path.name} and {gt_json.name}. Aborting evaluation.")
            return

        results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Found {len(tasks)} common files to evaluate.")
        logger.info(f"Individual file reports will be saved to: {results_dir}")
        logger.info(f"Final summary report will be saved to: {summary_output_path}")
        
        summary_data_list = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(
                    evaluate_and_save_single_file_api, 
                    gen_p, gt_p, self.evaluator_configs, self.weights, results_dir, model_name
                ): gen_p.name
                for gen_p, gt_p in tasks
            }
            for future in as_completed(future_to_file):
                try:
                    summary_data = future.result()
                    if summary_data:
                        summary_data_list.append(summary_data)
                except Exception as e:
                    file_name = future_to_file[future]
                    logger.error(f"A critical error occurred while processing the future for file {file_name}: {e}", exc_info=True)

        self._generate_summary_report(
            summary_data_list, generation_dir, gt_json_path, str(summary_output_path), model_name
        )


def main():
    parser = argparse.ArgumentParser(description="A strict, multi-step code evaluator using Large Language Models.")
    parser.add_argument('--gen-dir', type=str, required=True, help="Path to the directory containing generated code files.")
    parser.add_argument('--gt-json', type=str, required=True, help="Path to the JSON file with Ground-Truth code paths.")
    parser.add_argument('--summary-json-path', type=str, required=True, help="Full output path for the final summary JSON report.")
    parser.add_argument('--details-dir', type=str, required=True, help="Directory to store detailed reports for each file.")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the API model to use (e.g., 'gpt-4o').")
    parser.add_argument('--workers', type=int, default=4, help="Number of worker processes for parallel execution.")
    args = parser.parse_args()

    evaluator = APIBasedCodeEvaluator()
    print("\n" + "="*30 + " Starting Strict Multi-Step API Evaluation " + "="*30)
    evaluator.evaluate_and_report(
        generation_dir=args.gen_dir,
        gt_json_path=args.gt_json,
        summary_json_path=args.summary_json_path,
        details_dir=args.details_dir,
        model_name=args.model_name,
        num_workers=args.workers
    )
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()

