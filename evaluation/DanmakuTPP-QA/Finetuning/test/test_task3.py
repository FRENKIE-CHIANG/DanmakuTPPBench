# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, # Or your specific model class
    BitsAndBytesConfig
)
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# User-provided chat template (from your training script)
USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Default base model path (same as in your training script)
DEFAULT_BASE_MODEL_PATH = "PATH_TO_YOUR_BASE_MODEL"
DEFAULT_PEFT_MODEL_PATH = "PATH_TO_YOUR_PEFT_MODEL_PATH"
DEFAULT_TEST_DATA_FILE = "PATH_TO_YOUR_TEST_DATA_FILE"
DEFAULT_OUTPUT_DIR = "PATH_TO_YOUR_OUTPUT_DIR"

def parse_args():
    parser = argparse.ArgumentParser(description="Test a fine-tuned Qwen VL model for task 3 (next comment time prediction) and calculate RMSE.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Path to the base Hugging Face model.")
    parser.add_argument("--peft_model_path", type=str, default="/root/autodl-tmp/results_danmaku_sft_vl_task4/Qwen2.5-VL-3B_danmaku_sft_vl/checkpoint-1503", help="Path to the trained PEFT (LoRA) model adapter (e.g., final_checkpoint directory).")
    parser.add_argument("--test_data_file", type=str, default="/root/autodl-tmp/MMTPP/huggingface_data/task_4_next_peak_time/test.json", help="Path to the JSON test data file for task 3.")
    parser.add_argument("--output_dir", type=str, default="./test_results_task4", help="Directory to save test results and logs.")
    parser.add_argument("--max_seq_length", type=int, default=1775, help="Max sequence length for tokenization.")
    parser.add_argument("--max_raw_comment_seq_len", type=int, default=201, help="Max raw comment sequence length for input prompt construction.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--use_qlora_for_base", action='store_true', help="If the base model was loaded with QLoRA for training, set this to load it similarly for inference (though typically not needed if merging).")
    return parser.parse_args()

def load_test_data(data_file_path, tokenizer, args):
    logger.info(f"Loading test data from {data_file_path}...")
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse JSON file: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        logger.error("Test data should be a list of dictionaries.")
        sys.exit(1)

    prompts = []
    ground_truths_val = [] # Store numerical ground truth values

    for item in tqdm(raw_data, desc="Preparing test prompts"):
        original_comment_sequence = item.get("comment_time_sequence", [])
        if args.max_raw_comment_seq_len != -1 and len(original_comment_sequence) > args.max_raw_comment_seq_len:
            truncated_comment_sequence = original_comment_sequence[-args.max_raw_comment_seq_len:]
        else:
            truncated_comment_sequence = original_comment_sequence

        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))
        question_str = item.get("question", "") # Should be "请预测下一个时间点" or similar for task3
        
        # Ground truth for task3 is the next comment time (a single number)
        gt_next_time = item.get("ground_truth") 
        if gt_next_time is None:
            logger.warning(f"Missing 'ground_truth' in item: {item}. Skipping this item for RMSE calculation.")
            continue
        try:
            ground_truths_val.append(float(gt_next_time))
        except ValueError:
            logger.warning(f"Could not convert ground_truth '{gt_next_time}' to float. Skipping item: {item}")
            continue

        user_content = f"弹幕时间序列: [{comment_seq_str}]\n\n问题: {question_str}"
        messages_for_prompt = [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": user_content}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages_for_prompt,
            tokenize=False,
            add_generation_prompt=True # Important to prompt the model for a response
        )
        prompts.append(prompt_text)

    if not prompts:
        logger.error("No valid test prompts could be generated. Check your test data file format and content.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(prompts)} test prompts.")
    return prompts, ground_truths_val

def extract_next_time_from_response(response_text):
    """
    Extracts the predicted next time value from the model's response string.
    Example response for task 3: "下一个时间点 123.45。"
    """
    # Try to find a number (integer or float) after "下一个时间点"
    match = re.search(r"下一个波峰\s*([+-]?\d*\.?\d+)", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            logger.warning(f"Could not parse number from matched group: {match.group(1)} in response: '{response_text}'")
            return None
    # else:
    #     # Fallback: try to find any number in the string if the specific pattern fails
    #     # This is less precise and might pick up other numbers if the response format is unexpected.
    #     numbers = re.findall(r"[+-]?\d*\.?\d+", response_text)
    #     if numbers:
    #         try:
    #             # Let's assume the first number found is the relevant one, or the most prominent one.
    #             # This might need adjustment based on observed model outputs.
    #             return float(numbers[0]) # Or numbers[-1] if it's typically at the end
    #         except ValueError:
    #             logger.warning(f"Could not parse number from general search: {numbers} in response: '{response_text}'")
    #             return None
    logger.warning(f"Could not extract next time value from response: '{response_text}'")
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = Path(args.output_dir) / "test_task4.log"
    logger.add(log_file_path)
    logger.info(f"Parsed arguments: {args}")

    logger.info(f"Loading base model from {args.base_model_path} and processor...")
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.chat_template = USER_CHAT_TEMPLATE # Apply the same chat template

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': "<|im_end|>"})
    
    quantization_config = None
    if args.use_qlora_for_base: # Typically for training, but if model was saved without merging
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA config enabled for loading base model.")

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not args.use_qlora_for_base and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    logger.info(f"Loading PEFT model from {args.peft_model_path}...")
    # Ensure peft_model_path points to the directory containing adapter_config.json and adapter_model.bin
    try:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
        model = model.merge_and_unload() # Merge LoRA weights for faster inference if not using QLoRA for base
        logger.info("PEFT model loaded and merged with base model.")
    except Exception as e:
        logger.error(f"Failed to load PEFT model: {e}")
        logger.error("Make sure --peft_model_path points to the directory containing the adapter files (e.g., 'final_checkpoint').")
        sys.exit(1)

    model.eval() # Set model to evaluation mode

    prompts, ground_truth_values = load_test_data(args.test_data_file, tokenizer, args)
    
    if len(prompts) != len(ground_truth_values):
        logger.error(f"Mismatch between number of prompts ({len(prompts)}) and ground truth values ({len(ground_truth_values)}). Exiting.")
        sys.exit(1)
    if not ground_truth_values:
        logger.error("No ground truth values available for RMSE calculation. Exiting.")
        sys.exit(1)

    predictions_val = []
    raw_predictions_text = []

    logger.info(f"Starting inference on {len(prompts)} test samples...")
    current_device = next(model.parameters()).device # Get the device the model is on

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating predictions"):
        batch_prompts = prompts[i:i+args.batch_size]
        
        # Tokenize prompts - Qwen VL processor handles image/video if they were part of the prompt structure
        # For task 3, it's text-only based on your training script.
        inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
        inputs = {k: v.to(current_device) for k, v in inputs.items()}


        with torch.no_grad():
            # Set max_new_tokens to a reasonable value for predicting a single number
            # The expected output "下一个时间点 X。" is short.
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20, # Adjust if predictions are longer/shorter
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode the generated tokens, skipping special tokens and the prompt
        # generated_ids shape is (batch_size, sequence_length)
        # inputs['input_ids'].shape[1] gives the length of the prompt tokens
        
        batch_responses = []
        for j in range(generated_ids.shape[0]):
            prompt_len = inputs['input_ids'][j].ne(tokenizer.pad_token_id).sum().item()
            response_ids = generated_ids[j][prompt_len:] # Get only the generated part
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            batch_responses.append(response_text)

        raw_predictions_text.extend(batch_responses)

        for response_text in batch_responses:
            pred_time = extract_next_time_from_response(response_text)
            if pred_time is not None:
                predictions_val.append(pred_time)
            else:
                # Handle cases where prediction couldn't be parsed
                # Option 1: Append a NaN or placeholder and filter later (might skew RMSE if not handled)
                # Option 2: Skip this sample for RMSE (will reduce N)
                # For now, we'll effectively skip it if len(predictions_val) < len(ground_truth_values)
                # and align them carefully later.
                logger.warning(f"Could not parse prediction for a sample. Raw response: '{response_text}'")


    # --- RMSE Calculation ---
    # Ensure we only compare valid pairs
    num_successful_parses = len(predictions_val)
    
    # We need to align ground_truth_values with successfully parsed predictions.
    # This assumes that if a prediction fails to parse, we should exclude that corresponding ground truth.
    # This is a simplification. A more robust way would be to track indices.
    
    aligned_ground_truths = []
    aligned_predictions = []
    
    gt_idx = 0
    pred_idx = 0
    
    # Create a temporary list of original indices for ground truths that had parsable predictions
    # This requires knowing which original ground truths corresponded to failed parses.
    # For simplicity in this script, if parsing fails, that pair is dropped.
    # This implicitly assumes prompts and predictions are processed in the same order.

    if num_successful_parses == 0:
        logger.error("No predictions could be successfully parsed into numerical values. Cannot calculate RMSE.")
        rmse = float('nan')
    elif num_successful_parses < len(ground_truth_values):
        logger.warning(f"Only {num_successful_parses} out of {len(ground_truth_values)} predictions could be parsed. RMSE will be based on these.")
        # This case needs careful handling. For now, we assume the first `num_successful_parses` ground truths correspond
        # if there were parsing failures interspersed. This is NOT robust if failures are not at the end.
        # A better way: store (original_index, pred_value) and (original_index, gt_value)
        # For this script, we'll proceed with the available `predictions_val` and slice `ground_truth_values`
        # This assumes that if a prediction failed, it's simply missing from `predictions_val`
        # and we should compare with the corresponding ground truths that *did* yield a prediction.
        
        # Rebuild ground_truth_values based on successful predictions.
        # This is tricky without knowing WHICH ground truths correspond to failed predictions.
        # The current `predictions_val` only contains successful parses.
        # The `ground_truth_values` contains ALL ground truths.

        # Let's assume for now that `extract_next_time_from_response` appends None for failures
        # and we filter them out *before* this point.
        # If extract_next_time_from_response doesn't append None, but just skips, then predictions_val
        # is shorter.

        # Let's refine: iterate through raw_predictions_text and build aligned lists
        final_ground_truths_for_rmse = []
        final_predictions_for_rmse = []

        if len(raw_predictions_text) != len(ground_truth_values):
            logger.error(f"CRITICAL: Mismatch in raw predictions ({len(raw_predictions_text)}) and ground truths ({len(ground_truth_values)}) before alignment. This indicates a bug in processing.")
            sys.exit(1)

        for i, raw_pred_text in enumerate(raw_predictions_text):
            pred_val = extract_next_time_from_response(raw_pred_text) # Re-parse here to align
            if pred_val is not None:
                final_predictions_for_rmse.append(pred_val)
                final_ground_truths_for_rmse.append(ground_truth_values[i])
        
        if not final_predictions_for_rmse:
            logger.error("No predictions could be successfully parsed after re-alignment. Cannot calculate RMSE.")
            rmse = float('nan')
        else:
            logger.info(f"Calculating RMSE based on {len(final_predictions_for_rmse)} successfully parsed prediction-groundtruth pairs.")
            rmse = np.sqrt(mean_squared_error(final_ground_truths_for_rmse, final_predictions_for_rmse))
            logger.info(f"RMSE for next comment time prediction: {rmse:.4f}")

    else: # num_successful_parses == len(ground_truth_values)
        logger.info(f"Calculating RMSE based on all {len(predictions_val)} samples.")
        rmse = np.sqrt(mean_squared_error(ground_truth_values, predictions_val))
        logger.info(f"RMSE for next comment time prediction: {rmse:.4f}")

    # Save results
    results_summary_path = Path(args.output_dir) / "test_summary_task4.json"
    summary = {
        "base_model_path": args.base_model_path,
        "peft_model_path": args.peft_model_path,
        "test_data_file": args.test_data_file,
        "num_test_samples_total": len(ground_truth_values),
        "num_samples_for_rmse": len(final_predictions_for_rmse) if 'final_predictions_for_rmse' in locals() and final_predictions_for_rmse else 0,
        "rmse_next_time": rmse if 'rmse' in locals() else None,
    }
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    logger.info(f"Test summary saved to {results_summary_path}")

    # Optional: Save all predictions and ground truths
    detailed_results_path = Path(args.output_dir) / "detailed_predictions_task4.jsonl"
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        for i in range(len(raw_predictions_text)):
            pred_val = extract_next_time_from_response(raw_predictions_text[i]) # re-parse for consistent output
            f.write(json.dumps({
                "prompt": prompts[i],
                "raw_model_response": raw_predictions_text[i],
                "predicted_next_time": pred_val,
                "ground_truth_next_time": ground_truth_values[i]
            }, ensure_ascii=False) + "\n")
    logger.info(f"Detailed predictions saved to {detailed_results_path}")

    logger.info("Testing finished.")

if __name__ == "__main__":
    main()