# This script reads tasks from a JSON config, uses a prompt template with two images (data and style)
# to generate Python code with the MiMo-VL-7B model, saves the result, and logs the process.
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
# Note: Ensure Qwen2_5_VLForConditionalGeneration is the correct class for your model from its trust_remote_code
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ===================================================================================
# Model Configuration
# ===================================================================================
# 1. Model ID on Hugging Face Hub (for 'hub' loading)
HUB_MODEL_ID = "XiaomiMiMo/MiMo-VL-7B-RL-2508" # Example Hub ID for MiMo-VL models

# 2. Path to the local model (for 'local' loading)
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'MiMo-VL-7B-RL-2508'))
# ===================================================================================

# --- Global Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Inference Parameters ---
INFERENCE_PARAMS = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_new_tokens": 8192,
    "do_sample": True
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
"""

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

def process_vision_info(messages):
    """Extracts image paths from a message structure."""
    image_inputs = []
    for msg in messages:
        if msg['role'] == 'user':
            for content in msg['content']:
                if content['type'] == 'image':
                    image_inputs.append(content['image'])
    return image_inputs, None

def generate_code_for_image(data_image_path: str, style_image_path: str, task_instruction_text: str, model, processor) -> str:
    """Generates code for a given task using the MiMo-VL model."""
    try:
        final_prompt = PROMPT_TEMPLATE.format(task_instructions=task_instruction_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "text", "text": "\n\nThis is the data source image:"},
                    {"type": "image", "image": data_image_path},
                    {"type": "text", "text": "\n\nThis is the reference style image:"},
                    {"type": "image", "image": style_image_path},
                    # The final instruction is added to reinforce the output format
                    {"type": "text", "text": "\n\nNow, using the data from the data source image and applying the style from the reference image according to the instructions, please generate the Python code. The output format MUST be STRICTLY as follows: ```python\n# Your Python code here to reproduce the image.```"}
                ],
            }
        ]
        
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_paths, _ = process_vision_info(messages)
        pil_images = [Image.open(p) for p in image_paths]
        
        inputs = processor(
            text=[text_prompt],
            images=pil_images,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **INFERENCE_PARAMS,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()
    
    except Exception as e:
        tqdm.write(f"     - Error in inference function for data '{os.path.basename(data_image_path)}' | Error: {e}")
        return None

# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using MiMo-VL.")
    parser.add_argument("--load_source", type=str, required=True, choices=['hub', 'local'], help="Source to load the model from.")
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
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_to_load,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(model_to_load, max_pixels=4096 * 28 * 28, trust_remote_code=True)
        
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

    print("Cleaning JSON keys...")
    cleaned_tasks = [{key.strip(): value for key, value in task_dict.items()} for task_dict in tasks]
    tasks = cleaned_tasks
    data_root = os.path.dirname(args.json_path)
    print(f"Found {len(tasks)} tasks to process from JSON...")

    # --- 6. Process Tasks in Batch ---
    processed_count = 0
    failure_count = 0

    try:
        with open(LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Task ID", "Status", "Full Instruction", "Raw Model Output", "Extracted Code"])

            for task in tqdm(tasks, desc="Processing Tasks"):
                task_id = task.get("task_id", f"unknown_task_{processed_count + failure_count}")

                # Map JSON keys to variables
                relative_style_path = task.get("input image")
                relative_data_path = task.get("input image_2")
                relative_instruction_path = task.get("instruction")
                gt_image_path = task.get("GT image")

                # Validate paths
                if not all([relative_style_path, relative_data_path, relative_instruction_path, gt_image_path]):
                    tqdm.write(f"     - Warning: Skipping task '{task_id}' due to missing required keys in JSON.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                # Construct absolute paths
                style_image_path = os.path.join(data_root, relative_style_path)
                data_image_path = os.path.join(data_root, relative_data_path)
                instruction_path = os.path.join(data_root, relative_instruction_path)
                
                # Create output filename based on GT image
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

                # Run inference
                raw_generated_text = generate_code_for_image(data_image_path, style_image_path, instruction_text, model, processor)

                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(clean_code)
                    csv_writer.writerow([task_id, "SUCCESS", PROMPT_TEMPLATE.format(task_instructions=instruction_text), raw_generated_text, clean_code])
                    processed_count += 1
                else:
                    failure_count += 1
                    tqdm.write(f"     - Failure: Inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", PROMPT_TEMPLATE.format(task_instructions=instruction_text), "N/A", "N/A"])
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


