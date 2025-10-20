# This script generates Python code for data visualization tasks using an OpenAI-compatible API.
# It reads tasks from a JSON file, processes data, instructions, and a style image,
# then saves the generated code and a detailed log file.
# It is designed to be executed via a shell script providing command-line arguments.
import os
from tqdm import tqdm
import re
import shutil
import csv
import argparse
import json
import pandas as pd
import io
import base64
import openai
from dotenv import load_dotenv

# ===================================================================================
# API Configuration (Loaded from .env file in main)
# ===================================================================================
# Environment variables expected in the .env file:
# OPENAI_API_URL: The base URL for the API endpoint.
# OPENAI_TIMEOUT: (Optional) The timeout in seconds for API calls (default: 600).
# OPENAI_MODEL: The model identifier to use for generation (e.g., gemini-2.5-pro).
# ===================================================================================

API_MODEL = 'gpt-5-mini'
# --- Inference Parameters ---
INFERENCE_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 8192,
}

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a Python developer proficient in data visualization, with expertise in using libraries such as Matplotlib, NetworkX, Seaborn, pandas, and others.
Your task is to generate Python code that creates a plot based on the provided data and instructions. You will be given specific instructions, data in text format (extracted from a data file), and a style reference image.

Here are the general requirements:
1. **Use Provided Data**: The data you need to plot is provided below in CSV format. If the original was an Excel file, each sheet is clearly marked. You should use libraries like pandas and io.StringIO to parse this CSV data.
2. **Style Replication**: Replicate the visual style (colors, markers, layout, fonts, etc.) from the 'style reference image'.
3. **Follow Instructions**: Adhere to the specific instructions provided for the task.
4. **Self-contained Code**: The Python code must be complete, executable, and self-contained. The data should be defined directly within the code (e.g., in a pandas DataFrame loaded from a string), without needing to read any external files.

---
**Specific Task Instructions:**
{task_instructions}
---
**Data from File (in CSV format):**
{excel_data_string}
---

Now, using the data provided above and applying the style from the reference image according to the instructions, please generate the Python code.
The output format must be strictly as follows:

```python
# Your Python code here to reproduce the image.
```"""

# --- Helper Functions ---
def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        tqdm.write(f"   - Error: Failed to encode image {os.path.basename(image_path)}: {e}")
        return None

def read_csv_to_string(csv_path: str) -> str:
    """Reads the entire content of a CSV file and returns it as a string."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        tqdm.write(f"   - Error: Failed to read CSV file '{os.path.basename(csv_path)}': {e}")
        return None

def read_excel_to_csv_string(excel_path: str) -> str:
    """Reads all sheets from an Excel file and converts them into a single CSV-formatted string."""
    try:
        xls = pd.ExcelFile(excel_path)
        csv_output = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            string_buffer = io.StringIO()
            df.to_csv(string_buffer, index=False)
            csv_output.append(f"Sheet: `{sheet_name}`\n")
            csv_output.append(string_buffer.getvalue())
            csv_output.append("\n")
        return "\n".join(csv_output)
    except Exception as e:
        tqdm.write(f"   - Error: Failed to read or convert Excel file '{os.path.basename(excel_path)}' to CSV: {e}")
        return None

def extract_python_code(raw_text: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    match = re.search(r"```python\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match: return match.group(1).strip()
    return ""

def generate_code(excel_data_str: str, style_image_path: str, task_instruction_text: str, client, API_MODEL) -> str:
    """Generates code using an OpenAI-compatible API."""
    try:
        final_prompt = PROMPT_TEMPLATE.format(task_instructions=task_instruction_text, excel_data_string=excel_data_str)
        image_base64 = encode_image_to_base64(style_image_path)
        if not image_base64:
            return None

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"}}
            ]
        }]

        response = client.chat.completions.create(
            model=API_MODEL,
            messages=messages,
            **INFERENCE_PARAMS
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        tqdm.write(f"   - Error: Exception in API call -> Task: {os.path.basename(style_image_path)} | Error: {e}")
        return None

# --- Main Batch Processing ---
def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Generate Python code from JSON tasks using an OpenAI-compatible API.")
    # The --load_source argument is accepted for compatibility with the shell script but is not used beyond validation.
    parser.add_argument("--load_source", type=str, required=True, choices=['hub', 'local', 'api'], help="Source for model/API.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file containing tasks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated Python code.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to save the output CSV log file.")
    args = parser.parse_args()

    # 2. Setup Output Directories
    GENERATED_CODE_DIR = args.output_dir
    FAILED_FILES_DIR = f"{GENERATED_CODE_DIR}_failed"
    LOG_CSV_PATH = args.log_path

    os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
    os.makedirs(FAILED_FILES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    
    # 3. Load Environment Variables and Configure API Client
    load_dotenv()
    try:
        client = openai.OpenAI(
            base_url=os.getenv('OPENAI_API_URL'),
            timeout=float(os.getenv('OPENAI_TIMEOUT', 600.0)),
        )
        if not API_MODEL or not client.base_url:
            raise ValueError("OPENAI_API_URL and OPENAI_MODEL must be set in the .env file.")
        print(f"✅ API client configured for model: {API_MODEL}")
    except Exception as e:
        print(f"❌ OpenAI API client configuration failed: {e}")
        exit(1)

    # 4. Load and Prepare Task Data from JSON
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

    # 5. Batch Process Tasks
    processed_count = 0
    failure_count = 0

    try:
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Task ID", "Status", "Instruction Text", "Data String", "Raw Model Output", "Extracted Code"])

            for task in tqdm(tasks, desc="Processing tasks"):
                task_id = task.get("task_id", f"unknown_task_{processed_count + failure_count}")

                relative_data_path = task.get("input excel")
                relative_image_path = task.get("input image")
                relative_instruction_path = task.get("instruction")
                gt_image_path = task.get("GT image")

                if not all([relative_data_path, relative_image_path, relative_instruction_path, gt_image_path]):
                    tqdm.write(f"   - Warning: Skipping task '{task_id}' due to missing required keys.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                data_path = os.path.join(data_root, relative_data_path)
                image_path = os.path.join(data_root, relative_image_path)
                instruction_path = os.path.join(data_root, relative_instruction_path)
                
                base_filename = os.path.basename(gt_image_path)
                filename_root, _ = os.path.splitext(base_filename)
                output_path = os.path.join(GENERATED_CODE_DIR, f"{filename_root}.py")

                if os.path.exists(output_path):
                    tqdm.write(f"   - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", "N/A", "N/A", "N/A", "N/A"])
                    continue

                try:
                    with open(instruction_path, 'r', encoding='utf-8') as f:
                        instruction_text = f.read().strip()
                    
                    data_string = None
                    if data_path.endswith('.csv'):
                        data_string = read_csv_to_string(data_path)
                    elif data_path.endswith('.xlsx'):
                        data_string = read_excel_to_csv_string(data_path)
                    else:
                        tqdm.write(f"   - Error: Unsupported data file format for {task_id}: {data_path}")

                    if data_string is None:
                        raise IOError("Failed to read data file or format not supported.")

                except IOError as e:
                    tqdm.write(f"   - Error: Failed to read input files for {task_id}: {e}")
                    csv_writer.writerow([task_id, "FAILURE_READ_INPUT", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                raw_generated_text = generate_code(data_string, image_path, instruction_text, client, API_MODEL)

                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(clean_code)
                    csv_writer.writerow([task_id, "SUCCESS", instruction_text, data_string, raw_generated_text, clean_code])
                    processed_count += 1
                else:
                    failure_count += 1
                    tqdm.write(f"   - Failure: Inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", instruction_text, data_string, "N/A", "N/A"])
                    try:
                        shutil.copy(data_path, FAILED_FILES_DIR)
                        shutil.copy(image_path, FAILED_FILES_DIR)
                        shutil.copy(instruction_path, FAILED_FILES_DIR)
                    except Exception as e:
                        tqdm.write(f"   - Critical Error: Failed to copy failure files for '{task_id}': {e}")

    except IOError as e:
        print(f"❌ Critical Error: Could not write to CSV log file '{LOG_CSV_PATH}'. Error: {e}")
    finally:
        print(f"\n--- ✅ Batch Processing Complete (Model: {API_MODEL}) ---")
        print(f"Log saved to: {LOG_CSV_PATH}")
        print(f"Success: {processed_count} | Failed: {failure_count}")

if __name__ == "__main__":
    main()

