# This script reads sets of files (instruction, data, image) based on a JSON config,
# uses a dynamically generated prompt with the Qwen2.5-VL model to generate Python code,
# saves the result to a specified directory, and logs the process.
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Ensure the utility file is available
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("❌ Error: 'qwen_vl_utils.py' not found. Please ensure it is in the same directory.")
    exit(1)

# ===================================================================================
# Model Configuration
# ===================================================================================
# 1. Model ID on Hugging Face Hub (for 'hub' loading)
HUB_MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

# 2. Path to the local model (for 'local' loading)
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'Qwen2.5-VL-72B-Instruct'))
# ===================================================================================


# --- Global Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Inference Parameters ---
INFERENCE_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "max_new_tokens": 32768,
    "do_sample": True
}

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
def generate_code_for_image(image_path: str, instruction_text: str, data_text: str, model, processor) -> str:
    """Generates code for a given image using a dynamic prompt built from instruction and data."""
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

Now, based on the instruction, the data, and the reference image below, please generate the Python code. The output format must be strictly as follows:

```python
# Your Python code here to reproduce the plot.
"""

        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **INFERENCE_PARAMS)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response.strip()

    except Exception as e:
        tqdm.write(f"      - Error in inference function -> {os.path.basename(image_path)} | Error: {e}")
        return None


# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using Qwen 2.5 VL.")
    parser.add_argument("--load_source", type=str, required=True, choices=['hub', 'local'], help="Source to load the model from: 'hub' or 'local'.")
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

    # --- 3. Determine Model Path ---
    if args.load_source == 'local':
        model_to_load = LOCAL_MODEL_PATH
        print(f"Mode: Loading model from local path: {model_to_load}")
        if not os.path.isdir(model_to_load):
                print(f"❌ Error: Local model path not found. Please check the path in the script.")
                exit(1)
    else: # 'hub'
        model_to_load = HUB_MODEL_ID
        print(f"Mode: Loading model from Hugging Face Hub: {model_to_load}")

    # --- 4. Load Model and Processor ---
    print(f"Current device: {DEVICE}")
    try:
        processor = AutoProcessor.from_pretrained(model_to_load, max_pixels=1280 * 28 * 28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_to_load, torch_dtype=torch.float16, device_map="balanced").eval()
        
        # Add robustness for models that might not have a pad_token set
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        
        print("✅ Model and processor loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        exit()

    # --- 5. Load and Prepare Task Data from JSON ---
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"❌ Error reading or parsing JSON file '{args.json_path}': {e}")
        return

    # Clean the keys in each task dictionary to remove leading/trailing whitespace.
    print("Cleaning JSON keys...")
    cleaned_tasks = []
    for task_dict in tasks:
        cleaned_dict = {key.strip(): value for key, value in task_dict.items()}
        cleaned_tasks.append(cleaned_dict)
    tasks = cleaned_tasks
    
    data_root = os.path.dirname(args.json_path)
    print(f"Found {len(tasks)} tasks to process from JSON...")

    # --- 6. Process Tasks in Batch ---
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
                gt_image_path = task.get("GT image") # Get the GT image path for naming

                # Validate that all required file paths were found in the JSON task
                if not all([relative_image_path, relative_instruction_path, relative_data_path, gt_image_path]):
                    tqdm.write(f"      - Warning: Skipping task '{task_id}' due to missing required keys in JSON (e.g., input image, instruction, input data, GT image).")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                # Construct output path based on GT image filename
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
                    with open(instruction_path, 'r', encoding='utf-8') as f: instruction_text = f.read().strip()
                    with open(data_path, 'r', encoding='utf-8') as f: data_text = f.read().strip()
                except IOError as e:
                    tqdm.write(f"      - Error reading input files for {task_id}: {e}")
                    csv_writer.writerow([task_id, "FAILURE_READ_INPUT", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                # Skip processing if the output file already exists
                if os.path.exists(output_path):
                    tqdm.write(f"      - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", instruction_text, data_text, "N/A", "N/A"])
                    continue

                # Run the main inference function
                raw_generated_text = generate_code_for_image(image_path, instruction_text, data_text, model, processor)
                
                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    if clean_code:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(clean_code)
                        csv_writer.writerow([task_id, "SUCCESS", instruction_text, data_text, raw_generated_text, clean_code])
                        processed_count += 1
                    else:
                        failure_count += 1
                        tqdm.write(f"      - Failure: Could not extract code for task '{task_id}'")
                        csv_writer.writerow([task_id, "FAILURE_EXTRACT", instruction_text, data_text, raw_generated_text, ""])
                        shutil.copy(image_path, FAILED_FILES_DIR)
                else:
                    failure_count += 1
                    tqdm.write(f"      - Failure: Inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", instruction_text, data_text, "N/A", "N/A"])
                    try:
                        shutil.copy(image_path, FAILED_FILES_DIR)
                    except Exception as e:
                        tqdm.write(f"      - Critical Error: Failed to copy file on error: {e}")

    except IOError as e:
        print(f"❌ Critical Error: Could not write to CSV log file '{LOG_CSV_PATH}'. Error: {e}")
    finally:
        # --- 7. Print Final Summary ---
        print(f"\n--- ✅ Batch processing complete ---")
        print(f"Logs: {LOG_CSV_PATH}")
        print(f"Succeeded: {processed_count} | Failed: {failure_count}")

if __name__ == "__main__":
    main()

