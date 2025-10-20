# -*- coding: utf-8 -*-
# This script reads sets of files (instruction, data, image) based on a JSON config,
# uses the Doubao model via the Ark API to generate Python code,
# saves the result to a specified directory, and logs the process.
# It is designed to be run from a shell script that provides command-line arguments.

import os
import re
import shutil
import csv
import base64
import argparse
import json
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv 

# ===================================================================================
# Model and API Configuration
# ===================================================================================
# 1. Model ID for the Ark API
API_MODEL_ID = "doubao-1-5-thinking-vision-pro-250428"

# 2. Note on API Key and URL:
# The ARK_API_KEY and ARK_BASE_URL are not set here. They MUST be present
# in a .env file in the project's root directory.
# ===================================================================================


# --- Global Settings ---
# Inference parameters for the Ark API call
INFERENCE_PARAMS = {
    "max_tokens": 16384,
    "temperature": 0.1,
    "top_p": 0.9,
}


# --- Image Encoding Function ---
def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        tqdm.write(f"      - Error encoding image {image_path}: {e}")
        return None


# --- Code Extraction Function ---
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
def generate_code_for_image(image_path: str, instruction_text: str, data_text: str, client) -> str:
    """Generates code for a given image using the Ark API."""
    try:
        prompt = f"""You are a Python developer proficient in data visualization, with expertise in using libraries such as Matplotlib, NetworkX, Seaborn, and others. Your task is to generate Python code that can perfectly reproduce a plot based on a reference image, a natural language instruction, and the corresponding data.

Here are the requirements for the task:
1. **Use Provided Data**: You must use the data provided below in the generated code. Do not infer data from the image.
2. **Follow Instructions**: Adhere to the specific plotting instructions provided.
3. **Match Reference Image Style**: Use the reference image to understand the required visual style (colors, markers, line styles, labels, titles, legends, etc.) and replicate it as closely as possible.
4. **Self-contained Code**: The Python code should be complete, executable, and self-contained. It should not require any external data files. All data must be included within the script.

**Instruction:**
{instruction_text}

**Data:**
{data_text}
Now, based on the instruction, the data, and the reference image above, please generate the Python code. The output format must be strictly as follows:

```python
# Your Python code here to reproduce the plot.
"""
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model=API_MODEL_ID,
            messages=messages,
            **INFERENCE_PARAMS
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            tqdm.write(f"      - Warning: API returned an empty response for -> {os.path.basename(image_path)}")
            return None

    except Exception as e:
        tqdm.write(f"      - Error in inference function -> {os.path.basename(image_path)} | Error: {e}")
        return None


# --- Batch Processing Main Logic ---
def main():
    """Main execution function."""
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using the Ark API.")
    parser.add_argument("--load_source", type=str, required=True, choices=['hub', 'local'], help="Source to load the model from (ignored for API, but kept for compatibility).")
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

    # --- 3. Initialize API Client ---
    # <-- CHANGED: Entire block updated to use .env file -->
    load_dotenv()
    print(f"Mode: Using Ark API with model: {API_MODEL_ID}")
    
    api_key = os.environ.get("ARK_API_KEY")
    base_url = os.environ.get("ARK_BASE_URL")

    if not api_key:
        print("❌ Error: 'ARK_API_KEY' not found. Please add it to your .env file.")
        exit(1)
    if not base_url:
        print("❌ Error: 'ARK_BASE_URL' not found. Please add it to your .env file.")
        exit(1)
    
    try:
        client = Ark(base_url=base_url, api_key=api_key)
        print("✅ Ark API client initialized successfully!")
    except Exception as e:
        print(f"❌ API client initialization failed: {e}")
        exit(1)
    # <-- END OF CHANGES -->

    # --- 4. Load and Prepare Task Data from JSON ---
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"❌ Error reading or parsing JSON file '{args.json_path}': {e}")
        return
    
    print("Cleaning JSON keys...")
    cleaned_tasks = [{key.strip(): value for key, value in task_dict.items()} for task_dict in tasks]
    tasks = cleaned_tasks
    
    data_root = os.path.dirname(args.json_path)
    print(f"Found {len(tasks)} tasks to process from JSON...")

    # --- 5. Process Tasks in Batch ---
    succeeded_count = 0  # <-- CHANGED
    failure_count = 0    # <-- CHANGED
    skipped_count = 0    # <-- ADDED

    try:
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Task ID", "Status", "Instruction", "Data", "Raw Model Output", "Extracted Code"])

            for task in tqdm(tasks, desc="Processing Tasks"):
                task_id = task.get("task_id", f"unknown_task_{succeeded_count + failure_count + skipped_count}")
                
                relative_image_path = task.get("input image")
                relative_instruction_path = task.get("instruction")
                relative_data_path = task.get("input data")
                gt_image_path = task.get("GT image")

                if not all([relative_image_path, relative_instruction_path, relative_data_path, gt_image_path]):
                    tqdm.write(f"      - Warning: Skipping task '{task_id}' due to missing required keys in JSON.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                base_filename_with_ext = os.path.basename(gt_image_path)
                filename_root, _ = os.path.splitext(base_filename_with_ext)
                output_filename = f"{filename_root}.py"
                output_path = os.path.join(GENERATED_CODE_DIR, output_filename)

                image_path = os.path.join(data_root, relative_image_path)
                instruction_path = os.path.join(data_root, relative_instruction_path)
                data_path = os.path.join(data_root, relative_data_path)

                try:
                    with open(instruction_path, 'r', encoding='utf-8') as f: instruction_text = f.read().strip()
                    with open(data_path, 'r', encoding='utf-8') as f: data_text = f.read().strip()
                except IOError as e:
                    tqdm.write(f"      - Error reading input files for {task_id}: {e}")
                    csv_writer.writerow([task_id, "FAILURE_READ_INPUT", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                if os.path.exists(output_path):
                    tqdm.write(f"      - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", "N/A", "N/A", "N/A", "N/A"])
                    skipped_count += 1
                    continue

                raw_generated_text = generate_code_for_image(image_path, instruction_text, data_text, client)
                
                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    if clean_code:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(clean_code)
                        csv_writer.writerow([task_id, "SUCCESS", instruction_text, data_text, raw_generated_text, clean_code])
                        succeeded_count += 1 # <-- CHANGED
                    else:
                        tqdm.write(f"      - Failure: Could not extract code from model response for task '{task_id}'")
                        csv_writer.writerow([task_id, "FAILURE_EXTRACT", instruction_text, data_text, raw_generated_text, ""])
                        shutil.copy(image_path, FAILED_FILES_DIR)
                        failure_count += 1 # <-- CHANGED
                else:
                    tqdm.write(f"      - Failure: Inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", instruction_text, data_text, "N/A", "N/A"])
                    try:
                        shutil.copy(image_path, FAILED_FILES_DIR)
                    except Exception as e:
                        tqdm.write(f"      - Critical Error: Failed to copy file on error: {e}")
                    failure_count += 1 # <-- CHANGED

    except IOError as e:
        print(f"❌ Critical Error: Could not write to CSV log file '{LOG_CSV_PATH}'. Error: {e}")
    finally:
        # --- 6. Print Final Summary ---
        print(f"\n--- ✅ Batch processing complete ---")
        print(f"Logs: {LOG_CSV_PATH}")
        # <-- CHANGED: Updated summary to be more accurate -->
        print(f"Succeeded: {succeeded_count} | Failed: {failure_count} | Skipped: {skipped_count}")

if __name__ == "__main__":
    main()


