# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
from tqdm import tqdm
import re
import shutil
import csv
import argparse
import json
import base64
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv

# ===================================================================================
# API & Model Configuration
# ===================================================================================
# 1. Load environment variables for API key, URL, etc.
load_dotenv()

# 2. Specify the API model to be used for generation.
API_MODEL = "doubao-1-5-thinking-vision-pro-250428"
# ===================================================================================

# --- Global Settings ---
DEVICE = "N/A (API-based)"

# --- Inference Parameters (for API) ---
INFERENCE_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 8192,
    "top_p": 0.9,
    "thinking": {"type": "disabled"}
}

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are an expert Python developer specializing in data visualization with libraries like Matplotlib. I have an image of a plot and a set of instructions to modify it. Your task is to generate the Python code that would produce the *modified* plot.

Here are the requirements:
1. **Understand the Base Image**: Analyze the provided image to understand the original plot's data and structure.
2. **Apply Edits**: Carefully read the instructions provided below and apply them to the base plot.
3. **Generate Modified Code**: Generate a single, self-contained, and executable Python script that produces the final, edited visualization. The code should not require any external data files.

**Editing Instructions:**
---
{instructions}
---

Your objective is to generate a Python script that accurately reproduces the plot *after* applying the given instructions. The output format MUST be STRICTLY a Python code block.

```python
# Your Python code here to generate the MODIFIED image.
```"""

# --- Helper Functions ---
def extract_python_code(raw_text: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    match = re.search(r"```python\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        tqdm.write(f"     - Error encoding image {os.path.basename(image_path)}: {e}")
        return None

def generate_code_for_image(image_path: str, instruction_text: str, client) -> str:
    """Generates code via the Ark API."""
    try:
        full_prompt = PROMPT_TEMPLATE.format(instructions=instruction_text)
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            return None

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}
                }
            ]
        }]

        response = client.chat.completions.create(
            model=API_MODEL,
            messages=messages,
            **INFERENCE_PARAMS
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        tqdm.write(f"     - Error in API call for data '{os.path.basename(image_path)}' | Error: {e}")
        return None

# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using the Ark API.")
    parser.add_argument("--load_source", type=str, required=True, choices=['hub', 'local', 'api'], help="Source to load the model from (use 'api' for this script).")
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
    try:
        client = Ark(
            base_url=os.getenv('ARK_BASE_URL'),
            api_key=os.getenv('ARK_API_KEY')
        )
        print(f"✅ Ark API client configured for model: {API_MODEL}")
    except Exception as e:
        print(f"❌ Ark API client configuration failed: {e}")
        print("Please ensure ARK_BASE_URL and ARK_API_KEY are set in your .env file.")
        exit(1)

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
    processed_count = 0
    failure_count = 0

    try:
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Task ID", "Status", "Instruction Text", "Raw Model Output", "Extracted Code"])

            for task in tqdm(tasks, desc="Processing Tasks"):
                task_id = task.get("task_id", f"unknown_task_{processed_count + failure_count}")

                relative_image_path = task.get("input image")
                relative_instruction_path = task.get("instruction")
                gt_image_path = task.get("GT image")

                if not all([relative_image_path, relative_instruction_path, gt_image_path]):
                    tqdm.write(f"     - Warning: Skipping task '{task_id}' due to missing required keys in JSON.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                image_path = os.path.join(data_root, relative_image_path)
                instruction_path = os.path.join(data_root, relative_instruction_path)
                
                base_filename = os.path.basename(gt_image_path)
                filename_root, _ = os.path.splitext(base_filename)
                output_path = os.path.join(GENERATED_CODE_DIR, f"{filename_root}.py")

                if os.path.exists(output_path):
                    tqdm.write(f"     - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", "N/A", "N/A", "N/A"])
                    continue

                try:
                    with open(instruction_path, 'r', encoding='utf-8') as f:
                        instruction_text = f.read().strip()
                except IOError as e:
                    tqdm.write(f"     - Error reading instruction file for {task_id}: {e}")
                    csv_writer.writerow([task_id, "FAILURE_READ_INPUT", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                raw_generated_text = generate_code_for_image(image_path, instruction_text, client)

                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(clean_code)
                    csv_writer.writerow([task_id, "SUCCESS", instruction_text, raw_generated_text, clean_code])
                    processed_count += 1
                else:
                    failure_count += 1
                    tqdm.write(f"     - Failure: API call failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_API_CALL", instruction_text, "N/A", "N/A"])
                    try:
                        shutil.copy(image_path, FAILED_FILES_DIR)
                        shutil.copy(instruction_path, FAILED_FILES_DIR)
                    except Exception as e:
                        tqdm.write(f"     - Critical Error: Failed to copy files on error for '{task_id}': {e}")

    except IOError as e:
        print(f"❌ Critical Error: Could not write to CSV log file '{LOG_CSV_PATH}'. Error: {e}")
    finally:
        print(f"\n--- ✅ Batch processing complete ---")
        print(f"Logs: {LOG_CSV_PATH}")
        print(f"Succeeded: {processed_count} | Failed: {failure_count}")

if __name__ == "__main__":
    main()
