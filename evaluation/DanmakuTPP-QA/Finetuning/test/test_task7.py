# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset # Not strictly needed for this test script but kept for consistency
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, # Or your specific model class
    BitsAndBytesConfig
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score # For calculating accuracy

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
DEFAULT_PEFT_MODEL_PATH_TASK8 = "PATH_TO_YOUR_PEFT_MODEL_PATH"
DEFAULT_TEST_DATA_FILE_TASK8 = "PATH_TO_YOUR_TEST_DATA_FILE"
DEFAULT_OUTPUT_DIR_TASK8 = "PATH_TO_YOUR_OUTPUT_DIR"

def parse_args():
    parser = argparse.ArgumentParser(description="Test a fine-tuned Qwen VL model for task 8 (next comment event type prediction) and calculate accuracy.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Path to the base Hugging Face model.")
    parser.add_argument("--peft_model_path", type=str, default=DEFAULT_PEFT_MODEL_PATH_TASK8, help="Path to the trained PEFT (LoRA) model adapter for task 8 (e.g., final_checkpoint directory).")
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_TEST_DATA_FILE_TASK8, help="Path to the JSON test data file for task 8.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR_TASK8, help="Directory to save test results and logs for task 8.")
    parser.add_argument("--max_seq_length", type=int, default=8182, help="Max sequence length for tokenization.") # Consistent with task6 test
    parser.add_argument("--max_raw_seq_len", type=int, default=50, help="Max raw sequence length for input prompt construction (comment_time_sequence, event_type_sequence). From train_task8.py")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--use_qlora_for_base", action='store_true', help="If the base model was loaded with QLoRA for training, set this to load it similarly for inference.")
    return parser.parse_args()

def load_test_data(data_file_path, tokenizer, args):
    logger.info(f"Loading test data for task 8 from {data_file_path}...")
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
    ground_truths_event_type = [] # Store string ground truth event types

    for item in tqdm(raw_data, desc="Preparing task 8 test prompts"):
        # Process comment_time_sequence (from train_task8.py)
        original_comment_sequence = item.get("comment_time_sequence", [])
        if args.max_raw_seq_len != -1 and len(original_comment_sequence) > args.max_raw_seq_len:
            truncated_comment_sequence = original_comment_sequence[-args.max_raw_seq_len:]
        else:
            truncated_comment_sequence = original_comment_sequence
        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))

        # Process event_type_sequence (from train_task8.py)
        original_type_sequence = item.get("comment_sequence", [])
        if args.max_raw_seq_len != -1 and len(original_type_sequence) > args.max_raw_seq_len:
            truncated_type_sequence = original_type_sequence[-args.max_raw_seq_len:]
        else:
            truncated_type_sequence = original_type_sequence
        type_seq_str = ", ".join(map(str, truncated_type_sequence))

        question_str = item.get("question", "") # Should be "请预测下一个时间点的事件类型" or similar

        # Ground truth for task8 is the next event type (string)
        gt_event_type = item.get("ground_truth")
        if gt_event_type is None:
            logger.warning(f"Missing 'ground_truth' (event type) in item: {item}. Skipping this item.")
            continue
        ground_truths_event_type.append(str(gt_event_type)) # Ensure it's a string

        # Construct user_content as per train_task8.py
        user_content = f"弹幕时间序列: [{comment_seq_str}]\n\n弹幕文本序列: [{type_seq_str}]\n\n问题: {question_str}"
        messages_for_prompt = [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": user_content}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages_for_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt_text)

    if not prompts:
        logger.error("No valid test prompts could be generated for task 8. Check your test data file format and content.")
        sys.exit(1)

    logger.info(f"Loaded {len(prompts)} test prompts for task 8.")
    return prompts, ground_truths_event_type

def extract_event_type_from_response(response_text):
    """
    Extracts the predicted event type from the model's response string.
    Example response for task 8: "下一个时间点事件类型 高能。" or "下一个时间点事件类型 问号。"
    Assumes the event type is the text following "下一个时间点事件类型 " and before "。"
    """
    # Pattern: "下一个时间点事件类型 X。" (X is the event type)
    match = re.search(r"下一个时间点事件类型\s*(.+?)\s*。", response_text)
    if match:
        event_type = match.group(1).strip()
        if event_type: # Ensure extracted type is not empty
            return event_type
        else:
            logger.warning(f"Extracted event type is empty from response: '{response_text}'")
            return None
    else:
        # Fallback: Try to find common event type patterns if the main one fails.
        # This is highly dependent on your model's actual output variations.
        # Example: If the model sometimes just outputs the event type.
        # For now, we stick to the expected format.
        logger.warning(f"Could not extract event type using primary pattern from response: '{response_text}'")
        return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = Path(args.output_dir) / "test_task8.log" # Updated log file name
    logger.add(log_file_path)
    logger.info(f"Parsed arguments for task 8: {args}")

    logger.info(f"Loading base model from {args.base_model_path} and processor...")
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.chat_template = USER_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': "<|im_end|>"})

    quantization_config = None
    if args.use_qlora_for_base:
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

    logger.info(f"Loading PEFT model for task 8 from {args.peft_model_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
        model = model.merge_and_unload()
        logger.info("PEFT model loaded and merged with base model.")
    except Exception as e:
        logger.error(f"Failed to load PEFT model: {e}")
        logger.error("Make sure --peft_model_path points to the directory containing the adapter files (e.g., 'final_checkpoint').")
        sys.exit(1)

    model.eval()

    prompts, ground_truth_event_types = load_test_data(args.test_data_file, tokenizer, args)

    if len(prompts) != len(ground_truth_event_types):
        logger.error(f"Mismatch between number of prompts ({len(prompts)}) and ground truth event types ({len(ground_truth_event_types)}). Exiting.")
        sys.exit(1)
    if not ground_truth_event_types:
        logger.error("No ground truth event types available for accuracy calculation. Exiting.")
        sys.exit(1)

    raw_predictions_text = [] # To store all raw text outputs from the model

    logger.info(f"Starting inference on {len(prompts)} task 8 test samples...")
    current_device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating predictions for task 8"):
        batch_prompts = prompts[i:i+args.batch_size]
        inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
        inputs = {k: v.to(current_device) for k, v in inputs.items()}

        with torch.no_grad():
            # Max_new_tokens should be sufficient for "下一个时间点事件类型 X。"
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30, # Adjust if event types or output format are longer
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        batch_responses = []
        for j in range(generated_ids.shape[0]):
            prompt_len = inputs['input_ids'][j].ne(tokenizer.pad_token_id).sum().item()
            response_ids = generated_ids[j][prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            batch_responses.append(response_text)

        raw_predictions_text.extend(batch_responses)

    # --- Accuracy Calculation ---
    predicted_event_types_parsed = []
    corresponding_ground_truths_for_accuracy = []

    if len(raw_predictions_text) != len(ground_truth_event_types):
        logger.error(f"CRITICAL: Mismatch in raw predictions ({len(raw_predictions_text)}) and ground truths ({len(ground_truth_event_types)}) before alignment. This indicates a bug.")
        sys.exit(1)

    num_total_samples = len(ground_truth_event_types)
    num_valid_outputs = 0 # Samples where event type could be parsed

    for i, raw_pred_text in enumerate(raw_predictions_text):
        pred_event_type = extract_event_type_from_response(raw_pred_text)
        if pred_event_type is not None:
            predicted_event_types_parsed.append(pred_event_type)
            corresponding_ground_truths_for_accuracy.append(ground_truth_event_types[i])
            num_valid_outputs += 1
        else:
            logger.warning(f"Could not parse event type for sample {i}. Raw response: '{raw_pred_text}'. Ground truth: '{ground_truth_event_types[i]}'. This sample will be excluded from accuracy calculation.")

    event_type_accuracy = 0.0
    if not predicted_event_types_parsed: # No valid outputs parsed
        logger.error("No predicted event types could be successfully parsed. Accuracy is 0 or undefined.")
        event_type_accuracy = 0.0
    else:
        logger.info(f"Calculating accuracy based on {len(predicted_event_types_parsed)} successfully parsed prediction-groundtruth pairs (out of {num_total_samples} total samples).")
        event_type_accuracy = accuracy_score(corresponding_ground_truths_for_accuracy, predicted_event_types_parsed)
        logger.info(f"Event Type Prediction Accuracy: {event_type_accuracy:.4f}")

    # Save results
    results_summary_path = Path(args.output_dir) / "test_summary_task8.json" # Updated summary file name
    summary = {
        "base_model_path": args.base_model_path,
        "peft_model_path": args.peft_model_path,
        "test_data_file": args.test_data_file,
        "num_test_samples_total": num_total_samples,
        "num_valid_output_samples_for_accuracy": len(predicted_event_types_parsed),
        "event_type_accuracy": event_type_accuracy,
    }
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    logger.info(f"Test summary for task 8 saved to {results_summary_path}")

    detailed_results_path = Path(args.output_dir) / "detailed_predictions_task8.jsonl" # Updated detailed results file name
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        for i in range(len(raw_predictions_text)):
            # Re-parse for consistent output in the detailed file, or use a placeholder if parsing failed
            parsed_pred_for_log = extract_event_type_from_response(raw_predictions_text[i])
            f.write(json.dumps({
                "prompt_index": i,
                # "prompt": prompts[i], # Prompts can be very long, consider omitting or truncating if files become too large
                "raw_model_response": raw_predictions_text[i],
                "predicted_event_type": parsed_pred_for_log, # This will be None if parsing failed
                "ground_truth_event_type": ground_truth_event_types[i]
            }, ensure_ascii=False) + "\n")
    logger.info(f"Detailed predictions for task 8 saved to {detailed_results_path}")

    logger.info("Task 8 testing finished.")

if __name__ == "__main__":
    main()