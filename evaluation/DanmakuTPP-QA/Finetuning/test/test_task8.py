# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, # Or your specific model class
    BitsAndBytesConfig
)
from tqdm import tqdm
# sklearn.metrics.accuracy_score might not be directly applicable for the custom accuracy.

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
DEFAULT_PEFT_MODEL_PATH_TASK9 = "PATH_TO_YOUR_PEFT_MODEL_PATH"
DEFAULT_TEST_DATA_FILE_TASK9 = "PATH_TO_YOUR_TEST_DATA_FILE"
DEFAULT_OUTPUT_DIR_TASK9 = "PATH_TO_YOUR_OUTPUT_DIR"

def parse_args():
    parser = argparse.ArgumentParser(description="Test a fine-tuned Qwen VL model for task 9 (next peak event type prediction) and calculate custom accuracy.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Path to the base Hugging Face model.")
    parser.add_argument("--peft_model_path", type=str, default=DEFAULT_PEFT_MODEL_PATH_TASK9, help="Path to the trained PEFT (LoRA) model adapter for task 9 (e.g., final_checkpoint directory).")
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_TEST_DATA_FILE_TASK9, help="Path to the JSON test data file for task 9.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR_TASK9, help="Directory to save test results and logs for task 9.")
    parser.add_argument("--max_seq_length", type=int, default=8182, help="Max sequence length for tokenization.")
    # max_raw_seq_len comes from train_task9.py
    parser.add_argument("--max_raw_seq_len", type=int, default=100, help="Max raw sequence length for input prompt construction (comment_time_sequence, event_type_sequence). From train_task9.py")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--use_qlora_for_base", action='store_true', help="If the base model was loaded with QLoRA for training, set this to load it similarly for inference.")
    return parser.parse_args()

def load_test_data(data_file_path, tokenizer, args):
    logger.info(f"Loading test data for task 9 from {data_file_path}...")
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
    ground_truths_peak_events = [] # Store list of ground truth event strings

    for item in tqdm(raw_data, desc="Preparing task 9 test prompts"):
        # Process comment_time_sequence (from train_task9.py)
        original_comment_sequence = item.get("comment_time_sequence", [])
        if args.max_raw_seq_len != -1 and len(original_comment_sequence) > args.max_raw_seq_len:
            truncated_comment_sequence = original_comment_sequence[-args.max_raw_seq_len:]
        else:
            truncated_comment_sequence = original_comment_sequence
        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))

        # Process event_type_sequence (from train_task9.py)
        original_type_sequence = item.get("comment_sequence", [])
        if args.max_raw_seq_len != -1 and len(original_type_sequence) > args.max_raw_seq_len:
            truncated_type_sequence = original_type_sequence[-args.max_raw_seq_len:]
        else:
            truncated_type_sequence = original_type_sequence
        type_seq_str = ", ".join(map(str, truncated_type_sequence))

        next_peak_time_str = str(item.get("next_peak_time", "")) # From train_task9.py
        # Question specific to task 9, from train_task9.py
        # The train script uses: f"弹幕时间序列: [{comment_seq_str}]\n\n弹幕类型序列: [{type_seq_str}]\n\n下一个波峰点: {next_peak_time}\n\n问题: {question_str}"
        # And the question in the data is "请预测下一个波峰点的主要事件类型"
        question_str = item.get("question", "请预测下一个波峰点的主要事件类型")


        # Ground truth for task9 is a list of event strings
        gt_peak_events = item.get("ground_truth")
        if not isinstance(gt_peak_events, list) or not all(isinstance(event, str) for event in gt_peak_events):
            logger.warning(f"Missing or invalid 'ground_truth' (list of event strings) in item: {item}. Skipping this item.")
            continue
        ground_truths_peak_events.append(sorted(gt_peak_events)) # Store sorted for easier comparison later

        # Construct user_content as per train_task9.py
        user_content = f"弹幕时间序列: [{comment_seq_str}]\n\n弹幕文本序列: [{type_seq_str}]\n\n下一个波峰点: {next_peak_time_str}\n\n问题: {question_str}"

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
        logger.error("No valid test prompts could be generated for task 9. Check your test data file format and content.")
        sys.exit(1)

    logger.info(f"Loaded {len(prompts)} test prompts for task 9.")
    return prompts, ground_truths_peak_events

def extract_peak_events_from_response(response_text):
    """
    Extracts the predicted list of event types from the model's response string for task 9.
    Example response for task 9: "下一个波峰点主要事件类型 [social_interaction, humor/meme]。"
    """
    # Pattern: "下一个波峰点主要事件类型 [{event1}, {event2}, ...]。"
    match = re.search(r"下一个波峰点主要事件类型\s*\[([^\]]+)\]\s*。", response_text)
    if match:
        events_str = match.group(1).strip()
        if not events_str: # Handle cases like "[]"
            return []
        # Split by comma and strip whitespace from each event
        events = [event.strip() for event in events_str.split(',')]
        # Filter out any empty strings that might result from multiple commas or trailing commas
        events = [event for event in events if event]
        return sorted(events) # Return sorted for consistent comparison
    else:
        logger.warning(f"Could not extract peak events using primary pattern from response: '{response_text}'")
        return None

def calculate_custom_accuracy(predicted_events_list, ground_truth_events_list):
    """
    Calculates custom accuracy based on matching predicted events with ground truth events.
    - Ground truth is a list of 1 or 2 events.
    - Prediction is a list of 1 or 2 events (ideally, but can be more or less).

    Scoring:
    - If prediction is None (parsing failed), score is 0 for that sample.
    - If ground_truth has 1 event:
        - If prediction contains that 1 event: score 1.0
        - Else: score 0.0
    - If ground_truth has 2 events:
        - Convert prediction and ground_truth to sets for easier comparison.
        - Intersection size:
            - 2 (both match): score 1.0
            - 1 (one matches): score 0.5
            - 0 (none match): score 0.0
    """
    scores = []
    for i in range(len(predicted_events_list)):
        pred_events = predicted_events_list[i] # This is already sorted or None
        gt_events = ground_truth_events_list[i] # This is already sorted

        if pred_events is None: # Parsing failed for this prediction
            scores.append(0.0)
            continue

        # Ensure pred_events is a list, even if it's empty from parsing "[]"
        if not isinstance(pred_events, list):
             pred_events = [] # Should not happen if extract_peak_events_from_response returns list or None

        # Convert to sets for robust comparison
        pred_set = set(pred_events)
        gt_set = set(gt_events)

        # We expect ground truth to have 1 or 2 events as per task description.
        # Predictions ideally also have 1 or 2, but we handle general cases.
        # The problem description implies ground truth can have multiple events,
        # and the example shows two: ["social_interaction", "humor/meme"]

        intersection_size = len(pred_set.intersection(gt_set))
        
        if not gt_set: # Should not happen if data is correct
            scores.append(0.0) # Or handle as an error
            continue

        # Simpler logic:
        # Score 1 if all ground truth events are in predicted events AND all predicted events are in ground truth (perfect set match).
        # Score 0.5 if there's any overlap (intersection_size > 0) but not a perfect match.
        # Score 0 if no overlap.

        # The user's new rule:
        # "输出的和ground truth对上一个计为0.5，两个都对上计为1"
        # This implies ground truth usually has up to 2 items for this scoring.
        # Let's assume ground truth length is the target number of matches.

        if intersection_size == len(gt_set) and len(pred_set) == len(gt_set): # Perfect match of sets
            scores.append(1.0)
        elif intersection_size > 0: # At least one match
             # If GT has 1 event, intersection_size 1 means 1.0.
             # If GT has 2 events, intersection_size 1 means 0.5, intersection_size 2 means 1.0
            if len(gt_set) == 1 and intersection_size == 1:
                scores.append(1.0) # If GT has 1, and we matched it
            elif len(gt_set) == 2:
                if intersection_size == 2:
                     scores.append(1.0)
                elif intersection_size == 1:
                     scores.append(0.5)
                else: # intersection_size == 0
                     scores.append(0.0)
            elif intersection_size >= 1 : # Fallback for GT > 2 or if only partial match counts as 0.5
                 scores.append(0.5) # Generic partial match if gt_set rules above don't cover.
            else:
                 scores.append(0.0)

        else: # No overlap
            scores.append(0.0)
            
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = Path(args.output_dir) / "test_task9.log"
    logger.add(log_file_path)
    logger.info(f"Parsed arguments for task 9: {args}")

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

    logger.info(f"Loading PEFT model for task 9 from {args.peft_model_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
        model = model.merge_and_unload() 
        logger.info("PEFT model loaded and merged with base model.")
    except Exception as e:
        logger.error(f"Failed to load PEFT model: {e}")
        logger.error("Make sure --peft_model_path points to the directory containing the adapter files (e.g., 'final_checkpoint').")
        sys.exit(1)

    model.eval()

    prompts, ground_truth_peak_events_list = load_test_data(args.test_data_file, tokenizer, args)

    if len(prompts) != len(ground_truth_peak_events_list):
        logger.error(f"Mismatch between number of prompts ({len(prompts)}) and ground truth event lists ({len(ground_truth_peak_events_list)}). Exiting.")
        sys.exit(1)
    if not ground_truth_peak_events_list:
        logger.error("No ground truth event lists available for accuracy calculation. Exiting.")
        sys.exit(1)

    raw_predictions_text = []

    logger.info(f"Starting inference on {len(prompts)} task 9 test samples...")
    current_device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating predictions for task 9"):
        batch_prompts = prompts[i:i+args.batch_size]
        inputs = processor(text=batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_length)
        inputs = {k: v.to(current_device) for k, v in inputs.items()}

        with torch.no_grad():
            # Max_new_tokens should be sufficient for "下一个波峰点主要事件类型 [event1, event2]。"
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50, # Adjust if event lists or output format are longer
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
    # Store lists of predicted events (or None if parsing fails)
    predicted_peak_events_parsed_list = []

    if len(raw_predictions_text) != len(ground_truth_peak_events_list):
        logger.error(f"CRITICAL: Mismatch in raw predictions ({len(raw_predictions_text)}) and ground truths ({len(ground_truth_peak_events_list)}) before alignment. This indicates a bug.")
        sys.exit(1)

    num_total_samples = len(ground_truth_peak_events_list)
    
    for i, raw_pred_text in enumerate(raw_predictions_text):
        pred_events = extract_peak_events_from_response(raw_pred_text) # Returns sorted list or None
        predicted_peak_events_parsed_list.append(pred_events) # Appending the list of events or None
        if pred_events is None:
            logger.warning(f"Could not parse peak events for sample {i}. Raw response: '{raw_pred_text}'. Ground truth: {ground_truth_peak_events_list[i]}.")

    peak_event_accuracy = calculate_custom_accuracy(predicted_peak_events_parsed_list, ground_truth_peak_events_list)
    
    num_successfully_parsed = sum(1 for p in predicted_peak_events_parsed_list if p is not None)
    logger.info(f"Calculated custom accuracy based on {num_successfully_parsed} successfully parsed predictions out of {num_total_samples} total samples.")
    logger.info(f"Peak Event Prediction Custom Accuracy: {peak_event_accuracy:.4f}")

    # Save results
    results_summary_path = Path(args.output_dir) / "test_summary_task9.json"
    summary = {
        "base_model_path": args.base_model_path,
        "peft_model_path": args.peft_model_path,
        "test_data_file": args.test_data_file,
        "num_test_samples_total": num_total_samples,
        "num_successfully_parsed_predictions": num_successfully_parsed,
        "peak_event_custom_accuracy": peak_event_accuracy,
    }
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    logger.info(f"Test summary for task 9 saved to {results_summary_path}")

    detailed_results_path = Path(args.output_dir) / "detailed_predictions_task9.jsonl"
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        for i in range(len(raw_predictions_text)):
            # predicted_peak_events_parsed_list already contains the parsed events or None
            parsed_pred_for_log = predicted_peak_events_parsed_list[i]
            f.write(json.dumps({
                "prompt_index": i,
                # "prompt": prompts[i], # Prompts can be very long
                "raw_model_response": raw_predictions_text[i],
                "predicted_peak_events": parsed_pred_for_log, # This will be None or a list of strings
                "ground_truth_peak_events": ground_truth_peak_events_list[i] # This is a list of strings
            }, ensure_ascii=False) + "\n")
    logger.info(f"Detailed predictions for task 9 saved to {detailed_results_path}")

    logger.info("Task 9 testing finished.")

if __name__ == "__main__":
    main()