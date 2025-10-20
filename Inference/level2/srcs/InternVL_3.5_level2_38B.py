# Description:
# This script reads tasks from a JSON config, uses a prompt template with a base image
# and edit instructions to generate Python code with the InternVL model, saves the result, and logs the process.
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
import math
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ===================================================================================
# Model Configuration
# ===================================================================================
# 1. Model ID on Hugging Face Hub (for 'hub' loading)
HUB_MODEL_ID = "OpenGVLab/InternVL3_5-38B"

# 2. Path to the local model (for 'local' loading)
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'InternVL3_5-38B'))
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

Your objective is to generate a Python script that accurately reproduces the plot *after* applying the given instructions. The output format must be strictly a Python code block.

```python
# Your Python code here to generate the MODIFIED image.
```"""

# --- Helper Functions ---

# --- InternVL Specific Image Preprocessing Helpers ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float('inf'), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff, best_ratio = ratio_diff, ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    ), key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def process_image_internvl(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values

# --- General Helpers ---
def extract_python_code(raw_text: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    match = re.search(r"```python\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def generate_code_for_image(image_path: str, instruction_text: str, model, tokenizer) -> str:
    """Generates code for a given image using edit instructions."""
    try:
        full_prompt = PROMPT_TEMPLATE.format(instructions=instruction_text)
        image = Image.open(image_path).convert('RGB')
        pixel_values = process_image_internvl(image).to(torch.float16).cuda()
        
        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, full_prompt, INFERENCE_PARAMS)
        return response.strip()

    except Exception as e:
        tqdm.write(f"     - Error in inference function -> {os.path.basename(image_path)} | Error: {e}")
        return None

# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using InternVL.")
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

    # --- 4. Load Model and Tokenizer ---
    print(f"Current device: {DEVICE}")
    try:
        model = AutoModel.from_pretrained(
            model_to_load,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="balanced",
            trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True, use_fast=False)
        print("✅ InternVL model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        exit(1)

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

                raw_generated_text = generate_code_for_image(image_path, instruction_text, model, tokenizer)

                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(clean_code)
                    csv_writer.writerow([task_id, "SUCCESS", instruction_text, raw_generated_text, clean_code])
                    processed_count += 1
                else:
                    failure_count += 1
                    tqdm.write(f"     - Failure: Inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", instruction_text, "N/A", "N/A"])
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

