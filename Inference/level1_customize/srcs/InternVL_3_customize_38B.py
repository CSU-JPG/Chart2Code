# This script reads sets of files (instruction, data, image) based on a JSON config,
# uses a dynamically generated prompt with the InternVL3-38B model to generate Python code,
# saves the result to a specified directory, and logs the process.
# It is designed to be run from a shell script that provides command-line arguments.

import os
import torch
import re
import shutil
import csv
import argparse
import json
import math
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ===================================================================================
# Model Configuration
# ===================================================================================
# 1. Model ID on Hugging Face Hub (for 'hub' loading)
#    Note: Update this if you use a different Hub model name.
HUB_MODEL_ID = "OpenGVLab/InternVL3-38B" # Example, update as needed

# 2. Path to the local model (for 'local' loading)
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'InternVL3-38B'))
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

# --- Model Specific Helpers ---
def get_device_map(model_path: str) -> dict:
    """Calculates a custom device map for splitting a large model across available GPUs."""
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size == 0:
        return "cpu"
        
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
    except Exception as e:
        print(f"❌ Warning: Could not read config from '{model_path}' to create device map. Error: {e}")
        print("Falling back to balanced device map.")
        return "balanced"

    # Custom splitting logic for the 38B model
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layer_distribution = [num_layers_per_gpu] * world_size
    layer_distribution[1] = math.ceil(layer_distribution[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(layer_distribution):
        for j in range(num_layer):
            if layer_cnt < num_layers:
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    
    # Ensure the last layer is on the primary device for compatibility, if not already.
    device_map[f'language_model.model.layers.{num_layers - 1}'] = world_size -1

    return device_map


# --- InternVL Image Preprocessing ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
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
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_image_internvl(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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
def generate_code_for_image(image_path: str, instruction_text: str, data_text: str, model, tokenizer) -> str:
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
        
        # Load and process the image using the InternVL method
        image = Image.open(image_path).convert('RGB')
        # The primary device for inputs is usually 'cuda:0' when using device_map
        pixel_values = process_image_internvl(image).to(torch.float16).to(DEVICE)
        
        # Generate response using model.chat with the dynamic prompt
        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, prompt, INFERENCE_PARAMS)
        
        return response.strip()

    except Exception as e:
        tqdm.write(f"     - Error in inference function -> {os.path.basename(image_path)} | Error: {e}")
        return None


# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file.")
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

    # --- 4. Load Model and Tokenizer ---
    print(f"Current device: {DEVICE}")
    try:
        print("Calculating custom device map for multi-GPU setup...")
        device_map = get_device_map(model_to_load)
        
        model = AutoModel.from_pretrained(
            model_to_load,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map).eval()
            
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True, use_fast=False)
        print("✅ Model and tokenizer loaded successfully!")
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
                
                relative_image_path = task.get("input image")
                relative_instruction_path = task.get("instruction")
                relative_data_path = task.get("input data")
                gt_image_path = task.get("GT image")

                if not all([relative_image_path, relative_instruction_path, relative_data_path, gt_image_path]):
                    tqdm.write(f"     - Warning: Skipping task '{task_id}' due to missing required keys in JSON.")
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
                    tqdm.write(f"     - Error reading input files for {task_id}: {e}")
                    csv_writer.writerow([task_id, "FAILURE_READ_INPUT", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue

                if os.path.exists(output_path):
                    tqdm.write(f"     - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", "N/A", "N/A", "N/A", "N/A"])
                    continue

                raw_generated_text = generate_code_for_image(image_path, instruction_text, data_text, model, tokenizer)
                
                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    if clean_code:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(clean_code)
                        csv_writer.writerow([task_id, "SUCCESS", instruction_text, data_text, raw_generated_text, clean_code])
                        processed_count += 1
                    else:
                        failure_count += 1
                        tqdm.write(f"     - Failure: Could not extract code for task '{task_id}'")
                        csv_writer.writerow([task_id, "FAILURE_EXTRACT", instruction_text, data_text, raw_generated_text, ""])
                        try:
                           shutil.copy(image_path, FAILED_FILES_DIR)
                        except Exception as e:
                           tqdm.write(f"     - Critical Error: Failed to copy file on error: {e}")
                else:
                    failure_count += 1
                    tqdm.write(f"     - Failure: Inference failed for task '{task_id}'")
                    csv_writer.writerow([task_id, "FAILURE_INFERENCE", instruction_text, data_text, "N/A", "N/A"])
                    try:
                        shutil.copy(image_path, FAILED_FILES_DIR)
                    except Exception as e:
                        tqdm.write(f"     - Critical Error: Failed to copy file on error: {e}")

    except IOError as e:
        print(f"❌ Critical Error: Could not write to CSV log file '{LOG_CSV_PATH}'. Error: {e}")
    finally:
        # --- 7. Print Final Summary ---
        print(f"\n--- ✅ Batch processing complete ---")
        print(f"Logs: {LOG_CSV_PATH}")
        print(f"Succeeded: {processed_count} | Failed: {failure_count}")

if __name__ == "__main__":
    main()



