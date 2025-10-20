# -*- coding: utf-8 -*-
# This script reads sets of files (instruction, data, image) based on a JSON config,
# uses the GPT-5 API to generate Python code,
# saves the result to a specified directory, and logs the process.
# It is designed to be run from a shell script that provides command-line arguments.

import os
import sys
import base64
import shutil
import re
import csv
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import openai

# ===================================================================================
# API Configuration
# ===================================================================================
# Load environment variables from a .env file if it exists
load_dotenv()

# 1. API Model Name
API_MODEL = 'gpt-5'

# 2. Initialize the API Client
try:
    client = openai.OpenAI(
        base_url=os.getenv('OPENAI_API_URL'),
        timeout=float(os.getenv('OPENAI_TIMEOUT', 600.0)),
    )
    print(f"✅ OpenAI API client configured for model: {API_MODEL}")
except Exception as e:
    print(f"❌ OpenAI API client configuration failed: {e}")
    print("Please ensure your .env file is correctly set up with OPENAI_API_URL.")
    sys.exit(1)

# ===================================================================================

# --- Inference Parameters ---
INFERENCE_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 8192
}

# --- Helper Functions ---
def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image to a base64 string for API inclusion."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        tqdm.write(f"      - Error encoding image {os.path.basename(image_path)}: {e}")
        return None


def extract_python_code(raw_text: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    match = re.search(r"```python\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# --- Core Inference Function ---
def generate_code_for_image(image_path: str, instruction_text: str, data_text: str) -> str:
    """Generates code for a given image using the API and a dynamic prompt."""
    system_prompt = """You are a Python developer proficient in data visualization, with expertise in using libraries such as Matplotlib, NetworkX, Seaborn, and others. Your task is to generate Python code that can perfectly reproduce a plot based on a reference image, a natural language instruction, and the corresponding data.

Here are the requirements for the task:
1. **Use Provided Data**: You must use the data provided below in the generated code. Do not infer data from the image.
2. **Follow Instructions**: Adhere to the specific plotting instructions provided.
3. **Match Reference Image Style**: Use the reference image to understand the required visual style (colors, markers, line styles, labels, titles, legends, etc.) and replicate it as closely as possible.
4. **Self-contained Code**: The Python code should be complete, executable, and self-contained. It should not require any external data files. All data must be included within the script.

Your output format MUST be STRICTLY as follows:

```python
# Your Python code here to reproduce the plot.
"""
    user_prompt = f"""Instruction:
{instruction_text}

Data:
{data_text}

Now, based on the instruction, the data, and the reference image above, please generate the Python code."""

    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return None

        # Construct the message payload for the API
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        # Make the API call
        response = client.chat.completions.create(
            model=API_MODEL,
            messages=messages,
            **INFERENCE_PARAMS
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        tqdm.write(f"      - Error in API inference -> {os.path.basename(image_path)} | Error: {e}")
        return None


# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    # This script accepts --load_source for compatibility with the shell runner, but does not use it.
    parser = argparse.ArgumentParser(description=f"Generate Python code from a JSON task file using the {API_MODEL} API.")
    parser.add_argument("--load_source", type=str, required=True, choices=['hub', 'local'], help="Source to load the model from (ignored for API, but required for compatibility).")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file containing tasks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated Python code.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to save the output CSV log file.")
    args = parser.parse_args()

    # --- 2. Setup Output Directories ---
    GENERATED_CODE_DIR = args.output_dir
    FAILED_FILES_DIR = f"{GENERATED_CODE_DIR}_failed"
    LOG_CSV_PATH = args.log_path

    os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
    os.makedirs(FAILED_FILES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)

    # --- 3. Announce Mode ---
    print(f"Mode: Using {API_MODEL} API for code generation.")

    # --- 4. Load and Prepare Task Data from JSON ---
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"❌ Error reading or parsing JSON file '{args.json_path}': {e}")
        return

    # Clean the keys in each task dictionary to remove leading/trailing whitespace.
    print("Cleaning JSON keys...")
    cleaned_tasks = [{key.strip(): value for key, value in task_dict.items()} for task_dict in tasks]
    tasks = cleaned_tasks

    data_root = os.path.dirname(args.json_path)
    print(f"Found {len(tasks)} tasks to process from JSON...")

    # --- 5. Process Tasks in Batch ---
    processed_count = 0
    failure_count = 0

    try:
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Task ID", "Status", "Instruction", "Data", "Raw Model Output", "Extracted Code"])

            for task in tqdm(tasks, desc="Processing Tasks"):
                task_id = task.get("task_id", f"unknown_task_{processed_count + failure_count}")

                # Get all required paths from the task dictionary
                relative_image_path = task.get("input image")
                relative_instruction_path = task.get("instruction")
                relative_data_path = task.get("input data")
                gt_image_path = task.get("GT image")

                # Validate that all required file paths were found
                if not all([relative_image_path, relative_instruction_path, relative_data_path, gt_image_path]):
                    tqdm.write(f"      - Warning: Skipping task '{task_id}' due to missing required keys in JSON.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                # Construct output path based on GT image filename (as per reference script)
                base_filename_with_ext = os.path.basename(gt_image_path)
                filename_root, _ = os.path.splitext(base_filename_with_ext)
                output_filename = f"{filename_root}.py"
                output_path = os.path.join(GENERATED_CODE_DIR, output_filename)

                # Construct full paths to input files
                image_path = os.path.join(data_root, relative_image_path)
                instruction_path = os.path.join(data_root, relative_instruction_path)
                data_path = os.path.join(data_root, relative_data_path)

                # Read the instruction and data files
                try:
                    with open(instruction_path, 'r', encoding='utf-8') as f:
                        instruction_text = f.read().strip()
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data_text = f.read().strip()
                except IOError as e:
                    tqdm.write(f"      - Error reading input files for {task_id}: {e}")
                    csv_writer.writerow([task_id, "FAILURE_READ_INPUT", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                # Skip processing if the output file already exists
                if os.path.exists(output_path):
                    tqdm.write(f"      - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", "N/A", "N/A", "N/A", "N/A"])
                    continue

                # Run the main inference function
                raw_generated_text = generate_code_for_image(image_path, instruction_text, data_text)

                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    if clean_code:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(clean_code)
                        csv_writer.writerow([task_id, "SUCCESS", instruction_text, data_text, raw_generated_text, clean_code])
                        processed_count += 1
                    else:
                        failure_count += 1
                        tqdm.write(f"      - Failure: Could not extract code from API response for task '{task_id}'")
                        csv_writer.writerow([task_id, "FAILURE_NO_CODE", instruction_text, data_text, raw_generated_text, "N/A"])
                        try:
                            shutil.copy(image_path, FAILED_FILES_DIR)
                        except Exception as e:
                            tqdm.write(f"      - Critical Error: Failed to copy file on error: {e}")
                else:
                    failure_count += 1
                    tqdm.write(f"      - Failure: API inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", instruction_text, data_text, "N/A", "N/A"])
                    try:
                        shutil.copy(image_path, FAILED_FILES_DIR)
                    except Exception as e:
                        tqdm.write(f"      - Critical Error: Failed to copy file on error: {e}")

    except IOError as e:
        print(f"❌ Critical Error: Could not write to CSV log file '{LOG_CSV_PATH}'. Error: {e}")
    finally:
        # --- 6. Print Final Summary ---
        print(f"\n--- ✅ Batch processing complete ---")
        print(f"Logs: {LOG_CSV_PATH}")
        print(f"Succeeded: {processed_count} | Failed: {failure_count}")


# --- Entry Point ---
if __name__ == "__main__":
    main()

