import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Subprocess Runner Code (No changes here) ---
RUNNER_CODE = """
import sys
import os
import runpy
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for server-side execution
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from unittest.mock import patch

# --- Core Font Override Logic ---
# Get the absolute path to the font file from the main process.
font_path = sys.argv[3] if len(sys.argv) > 3 else 'None'

if font_path != 'None' and os.path.exists(font_path):
    try:
        # Step 1: Add the font to Matplotlib's font manager.
        fm.fontManager.addfont(font_path)
        
        # Step 2: Set the newly added font as the default for sans-serif.
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    except Exception as e:
        print(f"WARNING: Failed to set custom font {font_path}. Error: {e}", file=sys.stderr)

# Step 3: Ensure the minus sign is displayed correctly in plots.
plt.rcParams['axes.unicode_minus'] = False
# --- End of Font Logic ---

script_path = sys.argv[1]
output_path = sys.argv[2]

# This is a placeholder function to disable the savefig call within the target script.
# This ensures that we control the final image saving process.
def disabled_savefig(*args, **kwargs):
    pass

# Patch matplotlib.pyplot.savefig to prevent the executed script from saving its own figures.
with patch('matplotlib.pyplot.savefig', new=disabled_savefig):
    try:
        # Execute the target script.
        runpy.run_path(script_path, run_name='__main__')
    except Exception:
        # If the script fails, print the traceback to stderr and exit.
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

# After execution, check if any matplotlib figures were created.
if plt.get_fignums():
    try:
        # Save the figure to the specified output path.
        plt.savefig(output_path, dpi=150)
        plt.close('all')
        sys.exit(0) # Exit code 0 for success.
    except Exception as e:
        print(f"Error while saving figure: {e}", file=sys.stderr)
        plt.close('all')
        sys.exit(1) # Exit code 1 for error during saving.
else:
    # If the script ran but produced no figures, report it.
    print("Execution successful, but no matplotlib Figure was generated.", file=sys.stderr)
    plt.close('all')
    sys.exit(2) # Exit code 2 for success with no figure.
"""

def classify_error(error_message: str) -> str:
    """
    Classifies the error message into 'environment_error' or 'code_error'.
    """
    # Keywords that indicate a missing dependency or environment issue.
    env_error_keywords = [
        'ModuleNotFoundError', 'ImportError', 'No module named',
        'cannot import name', 'DLL load failed', 'package not found',
        'ModuleNotFound', 'module not found', 'import error',
        'Failed to import', 'could not import', 'Unable to import',
        'missing required dependencies', 'dependency not found'
    ]
    
    for keyword in env_error_keywords:
        if keyword.lower() in error_message.lower():
            return 'environment_error'
    
    # Defaults to code_error if no environment keywords are found.
    return 'code_error'

def process_single_script(script_path: Path, output_dir: Path, font_path: str):
    """
    Processes a single script in a separate subprocess.
    This is the target function for the ThreadPoolExecutor.
    """
    # Use the parent directory name as a prefix to create unique temporary file names.
    # This prevents conflicts if multiple scripts have the same name (e.g., 'main.py').
    prefix = script_path.parent.name
    temp_script_name = f"{prefix}_{script_path.name}"
    temp_image_stem = f"{prefix}_{script_path.stem}"
    
    # Define temporary paths for processing. Files will be moved to final paths only on success.
    temp_script_path = output_dir / temp_script_name
    temp_image_path = (output_dir / f"{temp_image_stem}.png").resolve()
    
    # Define final destination paths.
    final_script_path = output_dir / script_path.name
    final_image_path = output_dir / f"{script_path.stem}.png"

    command = [
        sys.executable, "-c", RUNNER_CODE,
        str(script_path), str(temp_image_path), str(font_path)
    ]
    
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            encoding='utf-8', timeout=60, cwd=script_path.parent
        )
        
        if result.returncode == 0: # Success
            if not temp_image_path.exists():
                return ('error', script_path.name, "Script claimed success but failed to generate the image file.")
            
            shutil.copy2(script_path, temp_script_path)
            warnings = result.stderr.strip()
            return ('success', script_path.name, warnings, temp_script_path, final_script_path, temp_image_path, final_image_path)
        
        elif result.returncode == 2: # Success, but no figure generated
            return ('error', script_path.name, result.stderr.strip() or "Script executed successfully but generated no Matplotlib figures.")

        else: # Script execution error
            return ('error', script_path.name, result.stderr.strip())

    except subprocess.TimeoutExpired:
        return ('error', script_path.name, "Script execution timed out (60 seconds).")
    except Exception as e:
        return ('error', script_path.name, f"An unknown error occurred while launching the subprocess: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Concurrently process Python scripts in two phases and safely rename result files."
    )
    parser.add_argument("script_files", type=str, nargs='+', help="A list of Python script files to process.")
    parser.add_argument("--output-dir", type=str, required=True, help="A single directory to store all successful results.")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent worker threads.")
    args = parser.parse_args()

    start_time = datetime.now()

    # --- Font File Detection ---
    script_dir = Path(__file__).parent.resolve()
    font_file = script_dir / "SimHei.ttf"
    font_path = "None"
    if font_file.exists():
        font_path = str(font_file)
        logger.info(f"Font file found. Will enforce its use: {font_path}")
    else:
        logger.warning(f"Warning: Font file 'SimHei.ttf' not found in {script_dir}. System default font will be used.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "execute_report.json"
    
    scripts_to_run = [Path(f) for f in args.script_files]
    total_scripts = len(scripts_to_run)
    logger.info(f"--- Phase 1: Concurrently processing {total_scripts} scripts in '{output_dir.name}' ---")

    warnings_info = []
    errors_info = []
    environment_errors = []
    code_errors = []
    rename_tasks = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_script = {
            executor.submit(process_single_script, script, output_dir, font_path): script
            for script in scripts_to_run
        }
        
        for future in as_completed(future_to_script):
            original_script = future_to_script[future]
            try:
                result = future.result()
                status = result[0]
                
                if status == 'success':
                    _, script_name, warnings, temp_script, final_script, temp_image, final_image = result
                    if warnings:
                        warnings_info.append({"file_name": script_name, "warnings": warnings})
                    rename_tasks.append({
                        "temp_script": temp_script, "final_script": final_script,
                        "temp_image": temp_image, "final_image": final_image
                    })
                elif status == 'error':
                    _, script_name, message = result
                    error_type = classify_error(message)
                    error_detail = {"file_name": script_name, "error_reason": message, "error_type": error_type}
                    errors_info.append(error_detail)
                    
                    if error_type == 'environment_error':
                        # --- NEW FEATURE: Log environment errors for immediate visibility ---
                        logger.info(f"Detected ENVIRONMENT ERROR for script: {script_name}")
                        environment_errors.append(error_detail)
                    else:
                        code_errors.append(error_detail)

            except Exception as exc:
                error_detail = {"file_name": original_script.name, "error_reason": f"An unexpected exception occurred during processing: {exc}", "error_type": "code_error"}
                errors_info.append(error_detail)
                code_errors.append(error_detail)
    
    logger.info("--- Phase 1 processing complete ---")

    # --- Phase 2: Move and overwrite files ---
    logger.info(f"--- Phase 2: Renaming {len(rename_tasks)} successful file pairs ---")
    for task in rename_tasks:
        shutil.move(str(task["temp_script"]), str(task["final_script"]))
        shutil.move(str(task["temp_image"]), str(task["final_image"]))
    logger.info("--- Phase 2 renaming complete ---")

    # Calculate final statistics
    success_count = len(rename_tasks)
    error_count = len(errors_info)
    pass_rate = (success_count / total_scripts * 100) if total_scripts > 0 else 0
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    # --- ROBUSTNESS ENHANCEMENT: Reordered JSON report for better readability ---
    report = {
        "execution_summary": {
            "execution_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(execution_time, 2),
            "total_scripts": total_scripts,
            "success_count": success_count,
            "error_count": error_count,
            "warning_count": len(warnings_info),
            "pass_rate": f"{pass_rate:.2f}%",
            "runnable_ratio": f"{success_count}/{total_scripts}"
        },
        "error_classification": {
            "environment_errors": {
                "count": len(environment_errors),
                "percentage": f"{(len(environment_errors) / error_count * 100):.2f}%" if error_count > 0 else "0.00%"
            },
            "code_errors": {
                "count": len(code_errors),
                "percentage": f"{(len(code_errors) / error_count * 100):.2f}%" if error_count > 0 else "0.00%"
            }
        },
        "warnings": warnings_info,
        "environment_errors_details": environment_errors,
        "code_errors_details": code_errors,
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Execution report saved to '{report_file}'.")
    logger.info(f"Processing Complete: Success: {success_count}, Warnings: {len(warnings_info)}, Errors: {error_count}")
    logger.info(f"Error Breakdown: Environment Errors: {len(environment_errors)}, Code Errors: {len(code_errors)}")
    logger.info(f"Pass Rate: {pass_rate:.2f}%, Runnable: {success_count}/{total_scripts}")
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

    sys.exit(0)

if __name__ == "__main__":
    main()
