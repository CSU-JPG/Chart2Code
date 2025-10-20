# This script reads tasks from a JSON config, uses a prompt template with two images
# (data and style) to generate Python code via the Ark API, saves the result, and logs the process.
# It is designed to be run from a shell script that provides command-line arguments.
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
    "thinking": {"type": "disabled"}  # Or "enabled", "auto"
}

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a Python developer proficient in data visualization, with expertise in using libraries such as Matplotlib, NetworkX, Seaborn, and others.
Your task is to generate Python code that reproduces a plot. You will be given specific instructions, a data source image, and a style reference image.

Here are the general requirements:
1. **Data Extraction**: Extract the necessary data from the 'data source image'.
2. **Style Replication**: Replicate the visual style (colors, markers, layout, etc.) from the 'style reference image'.
3. **Follow Instructions**: Adhere to the specific instructions provided for the task.
4. **Self-contained Code**: The Python code must be complete, executable, and self-contained, without needing external data files.

---
**Specific Task Instructions:**
{task_instructions}
---

Now, using the data from the data source image and applying the style from the reference image according to the instructions, please generate the Python code.
The output format must be strictly as follows:

```python
# Your Python code here to reproduce the image.
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

def generate_code_for_image(data_image_path: str, style_image_path: str, task_instruction_text: str, client, processor=None) -> str:
    """Generates code via the Ark API."""
    try:
        final_prompt = PROMPT_TEMPLATE.format(task_instructions=task_instruction_text)

        data_image_base64 = encode_image_to_base64(data_image_path)
        style_image_base64 = encode_image_to_base64(style_image_path)
        if not data_image_base64 or not style_image_base64:
            return None

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "text", "text": "\n\nThis is the data source image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data_image_base64}", "detail": "high"}},
                {"type": "text", "text": "\n\nThis is the style reference image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{style_image_base64}", "detail": "high"}},
            ]
        }]

        response = client.chat.completions.create(
            model=API_MODEL,
            messages=messages,
            **INFERENCE_PARAMS
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        tqdm.write(f"     - Error in API call for data '{os.path.basename(data_image_path)}' | Error: {e}")
        return None

# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using an Ark API.")
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
            csv_writer.writerow(["Task ID", "Status", "Full Instruction", "Raw Model Output", "Extracted Code"])

            for task in tqdm(tasks, desc="Processing Tasks"):
                task_id = task.get("task_id", f"unknown_task_{processed_count + failure_count}")

                relative_style_path = task.get("input image")
                relative_data_path = task.get("input image_2")
                relative_instruction_path = task.get("instruction")
                gt_image_path = task.get("GT image")

                if not all([relative_style_path, relative_data_path, relative_instruction_path, gt_image_path]):
                    tqdm.write(f"     - Warning: Skipping task '{task_id}' due to missing required keys in JSON.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                style_image_path = os.path.join(data_root, relative_style_path)
                data_image_path = os.path.join(data_root, relative_data_path)
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

                raw_generated_text = generate_code_for_image(data_image_path, style_image_path, instruction_text, client)

                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(clean_code)
                    csv_writer.writerow([task_id, "SUCCESS", PROMPT_TEMPLATE.format(task_instructions=instruction_text), raw_generated_text, clean_code])
                    processed_count += 1
                else:
                    failure_count += 1
                    tqdm.write(f"     - Failure: API call failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_API_CALL", PROMPT_TEMPLATE.format(task_instructions=instruction_text), "N/A", "N/A"])
                    try:
                        shutil.copy(style_image_path, FAILED_FILES_DIR)
                        shutil.copy(data_image_path, FAILED_FILES_DIR)
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
