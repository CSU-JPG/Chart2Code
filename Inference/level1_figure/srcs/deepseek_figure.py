# This script reads tasks from a JSON config, uses a prompt template with two images
# (data and style) to generate Python code with the DeepSeek-VL model, saves the result, and logs the process.
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
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# ===================================================================================
# Model Configuration
# ===================================================================================
HUB_MODEL_ID = "deepseek-ai/deepseek-vl-7b-chat"
script_dir = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'models', 'deepseek-vl-7b-chat'))
# ===================================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INFERENCE_PARAMS = {
    "max_new_tokens": 8192,
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.9,
    "use_cache": True
}

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


def generate_code_for_image(data_image_path: str, style_image_path: str, task_instruction_text: str, model, processor) -> str:
    """Generates code by injecting task instructions into the main prompt template."""
    try:
        # 1. Format the final prompt with specific task instructions.
        final_prompt = PROMPT_TEMPLATE.format(task_instructions=task_instruction_text)

        # 2. Build the conversation format for DeepSeek-VL with multiple images.
        content_with_placeholders = (
            f"{final_prompt}\n\n"
            "This is the data source image: <image_placeholder>\n\n"
            "This is the reference style image: <image_placeholder>"
        )

        conversation = [
            {
                "role": "User",
                "content": content_with_placeholders,
                "images": [data_image_path, style_image_path]
            },
            {"role": "Assistant", "content": ""}
        ]

        # 3. Load images and prepare inputs using the processor.
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(model.device)

        with torch.no_grad():
            # 4. Run the image encoder to get image embeddings.
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

            # 5. Run the language model to generate a response.
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                **INFERENCE_PARAMS
            )

        # 6. Decode the generated tokens into text.
        response = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        tqdm.write(f"     - Error in inference function for data '{os.path.basename(data_image_path)}' | Error: {e}")
        return None


# --- Batch Processing Main Logic ---
def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate Python code from a JSON task file using DeepSeek-VL.")
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
            print("❌ Error: Local model path not found. Please check the path in the script.")
            exit(1)
    else:  # 'hub'
        model_to_load = HUB_MODEL_ID
        print(f"Mode: Loading model from Hugging Face Hub: {model_to_load}")

    # --- 4. Load Model and Processor ---
    print(f"Current device: {DEVICE}")
    try:
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_to_load, trust_remote_code=True)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            trust_remote_code=True,
        ).to(torch.bfloat16).to(DEVICE).eval()
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

