# This script reads sets of files (instruction, data, image) based on a JSON config,
# uses a dynamically generated prompt with the InternVL2.5-38B model to generate Python code,
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
import math
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ===================================================================================
# Model Configuration
# ===================================================================================
# 1. Model ID on Hugging Face Hub (for 'hub' loading)
HUB_MODEL_ID = "OpenGVLab/InternVL2_5-38B"

# 2. Path to the local model (for 'local' loading)
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'InternVL2_5-38B'))
# ===================================================================================


# --- Global Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Inference Parameters ---
INFERENCE_PARAMS = {
    "max_new_tokens": 32768,
    "do_sample": False,
}

# --- InternVL Specific: Multi-GPU Device Mapping ---
def split_model(model_name='InternVL2_5-38B'):
    """
    Generates a custom device_map to distribute the InternVL model across available GPUs.
    This is necessary for very large models that don't fit on a single GPU.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size == 0:
        return 'cpu'
    if world_size == 1:
        return 'auto'
        
    # Layer counts for different InternVL model sizes
    num_layers_map = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80
    }
    num_layers = num_layers_map.get(model_name)
    if not num_layers:
        raise ValueError(f"Unknown model name for device mapping: {model_name}")

    # The first GPU holds the ViT, so it gets fewer language layers.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layers_on_gpus = [num_layers_per_gpu] * world_size
    layers_on_gpus[0] = math.ceil(layers_on_gpus[0] * 0.5)
    
    layer_cnt = 0
    for i, num_layer in enumerate(layers_on_gpus):
        for _ in range(num_layer):
            if layer_cnt >= num_layers: break
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
            
    # Assign remaining components to the first GPU
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    if num_layers > 0:
      device_map[f'language_model.model.layers.{num_layers - 1}'] = world_size -1 
    
    print("Generated Custom Device Map:", device_map)
    return device_map


# --- InternVL Specific: Image Preprocessing Pipeline ---
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
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * target_aspect_ratio[0], image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * image_size, (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size, ((i // target_aspect_ratio[0]) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def process_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# --- Code Extraction Function ---
def extract_python_code(raw_text: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    match = re.search(r"```python\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", raw_text, re.DOTALL)
    if match: return match.group(1).strip()
    return ""

# --- Core Inference Function (Adapted for InternVL API) ---
def generate_code_for_image(image_path: str, instruction_text: str, data_text: str, model, tokenizer) -> str:
    """Generates code for an image using the specific InternVL input and chat format."""
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
        image = Image.open(image_path).convert('RGB')
        # Use the model's specific image processing pipeline
        pixel_values = process_image(image).to(torch.float16).to(model.device)

        with torch.no_grad():
            # Use the model's specific `.chat()` method for inference
            response = model.chat(tokenizer, pixel_values, prompt, INFERENCE_PARAMS)

        return response.strip()

    except Exception as e:
        tqdm.write(f"      - Error in inference function -> {os.path.basename(image_path)} | Error: {e}")
        return None


# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using InternVL.")
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

    # --- 4. Load Model and Processor (with custom device mapping) ---
    print(f"Current device: {DEVICE}")
    try:
        # Generate the custom device map for multi-GPU loading
        device_map = split_model(model_name='InternVL2_5-38B')
        
        model = AutoModel.from_pretrained(
            model_to_load,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True, use_fast=False)
        print("✅ InternVL model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        exit()

    # --- 5. Load and Prepare Task Data from JSON ---
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f: tasks = json.load(f)
    except Exception as e:
        print(f"❌ Error reading or parsing JSON file '{args.json_path}': {e}")
        return

    print("Cleaning JSON keys...")
    tasks = [{k.strip(): v for k, v in task.items()} for task in tasks]

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
                    tqdm.write(f"      - Warning: Skipping task '{task_id}' due to missing required keys.")
                    csv_writer.writerow([task_id, "FAILURE_MISSING_PATHS", "N/A", "N/A", "N/A", "N/A"])
                    failure_count += 1
                    continue
                
                output_path = os.path.join(GENERATED_CODE_DIR, f"{os.path.splitext(os.path.basename(gt_image_path))[0]}.py")
                
                if os.path.exists(output_path):
                    tqdm.write(f"      - Skipping: Output file already exists '{output_path}'")
                    csv_writer.writerow([task_id, "SKIPPED_EXISTS", "N/A", "N/A", "N/A", "N/A"])
                    continue
                
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
                
                raw_generated_text = generate_code_for_image(image_path, instruction_text, data_text, model, tokenizer)
                
                if raw_generated_text:
                    clean_code = extract_python_code(raw_generated_text)
                    if clean_code:
                        with open(output_path, 'w', encoding='utf-8') as f: f.write(clean_code)
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
        print(f"\n--- ✅ Batch processing complete ---")
        print(f"Logs: {LOG_CSV_PATH}")
        print(f"Succeeded: {processed_count} | Failed: {failure_count}")

if __name__ == "__main__":
    main()

