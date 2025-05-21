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
    Qwen2_5_VLForConditionalGeneration, # Use the specific model class from training
    BitsAndBytesConfig
)
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Chat template from your training script
USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Default paths - adjust these based on your environment
DEFAULT_BASE_MODEL_PATH = "PATH_TO_YOUR_BASE_MODEL" 
DEFAULT_PEFT_MODEL_PATH_TASK5 = "PATH_TO_YOUR_PEFT_MODEL_PATH"
DEFAULT_TEST_DATA_FILE_TASK5 = "PATH_TO_YOUR_TEST_DATA_FILE"
DEFAULT_OUTPUT_DIR_TASK5 = "PATH_TO_YOUR_OUTPUT_DIR"
DEFAULT_IMAGE_BASE_PATH = "PATH_TO_YOUR_IMAGE_BASE_PATH"

def parse_args():
    parser = argparse.ArgumentParser(description="Test the fine-tuned Qwen VL model on Task 5 (Peak Sentiment Polarity Prediction) using RMSE and MAE.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH, help="Path to the base Hugging Face model.")
    parser.add_argument("--peft_model_path", type=str, default=DEFAULT_PEFT_MODEL_PATH_TASK5, help="Path to the trained Task 5 PEFT (LoRA) adapter directory (e.g., final_checkpoint).")
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_TEST_DATA_FILE_TASK5, help="Path to the Task 5 JSON test data file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR_TASK5, help="Directory to save Task 5 test results and logs.")
    parser.add_argument("--image_base_path", type=str, default=DEFAULT_IMAGE_BASE_PATH, help="Base path to prepend to relative image paths in the JSON.")
    # Match sequence lengths and processing parameters with train_task5.py
    parser.add_argument("--max_seq_length", type=int, default=8182, help="Maximum sequence length for tokenization (match train_task5.py).")
    parser.add_argument("--max_raw_seq_len", type=int, default=50, help="Maximum raw sequence length for comments/types used in prompt (match train_task5.py).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (1 is often safer for VLM).")
    parser.add_argument("--use_qlora", action='store_true', help="Set this if the base model was trained using QLoRA, to load it similarly for inference.")
    # Image target size is not explicitly in train_task5.py args, but resize happens in preprocess.
    # Processor usually handles this, but explicit resize can be added if needed.
    # parser.add_argument("--image_target_size_w", type=int, default=64, help="Image target width for preprocessing.")
    # parser.add_argument("--image_target_size_h", type=int, default=64, help="Image target height for preprocessing.")
    return parser.parse_args()

def load_test_data_task5(data_file_path, args):
    """Loads Task 5 test data and prepares prompts consistent with train_task5.py."""
    logger.info(f"Loading Task 5 test data from {data_file_path}...")
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse JSON file: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        logger.error("Test data should be a list of dictionaries.")
        sys.exit(1)

    all_messages_for_prompt = []
    all_ground_truth_polarities = []
    all_resolved_image_paths = [] # Store lists of resolved image paths for each sample

    for item_idx, item in enumerate(tqdm(raw_data, desc="Preparing Task 5 test prompts")):
        # Align data extraction and truncation with train_task5.py's load_and_prepare_dataset
        original_comment_sequence = item.get("comment_time_sequence", [])
        comment_len = len(original_comment_sequence)
        if args.max_raw_seq_len != -1 and comment_len > args.max_raw_seq_len:
            truncated_comment_sequence = original_comment_sequence[-args.max_raw_seq_len:]
        else:
            truncated_comment_sequence = original_comment_sequence
        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))

        original_type_sequence = item.get("comment_sequence", [])
        type_len = len(original_type_sequence)
        if args.max_raw_seq_len != -1 and type_len > args.max_raw_seq_len:
            truncated_type_sequence = original_type_sequence[-args.max_raw_seq_len:]
        else:
            truncated_type_sequence = original_type_sequence
        type_seq_str = ", ".join(map(str, truncated_type_sequence))

        # Get the question field used in training
        question_str = item.get("question", "") # Often asks for peak sentiment polarity in Task 5 training data

        # Task 5 ground truth is the polarity score
        gt_polarity = item.get("ground_truth")
        if gt_polarity is None:
            logger.warning(f"Item {item_idx} (ID: {item.get('id', 'N/A')}) missing 'ground_truth' (polarity). Skipping.")
            continue
        try:
            # Ground truth needs to be float for metric calculation
            all_ground_truth_polarities.append(float(gt_polarity))
        except (ValueError, TypeError):
            logger.warning(f"Item {item_idx} (ID: {item.get('id', 'N/A')}) 'ground_truth' ('{gt_polarity}') cannot be converted to float. Skipping.")
            continue

        # Process image paths, resolving against base path if needed
        image_paths_relative = item.get("video_frames_in_peak_time_window", [])
        resolved_image_paths_for_item = []
        image_base_dir = Path(args.image_base_path)
        if image_paths_relative:
            for rel_path in image_paths_relative:
                full_path = Path(rel_path)
                if args.image_base_path and not full_path.is_absolute():
                    resolved_image_paths_for_item.append(str(image_base_dir / rel_path))
                else:
                    resolved_image_paths_for_item.append(str(full_path)) # Store as string
        all_resolved_image_paths.append(resolved_image_paths_for_item) # Add list (even if empty)

        # Build messages structure exactly like in train_task5.py
        # System message first
        system_content_list = [{"type": "text", "text": "You are a helpful assistant."}] # Consistent with training
        # User message content: images first, then text
        user_content_list = []
        # Add image placeholders based on resolved paths
        for _ in resolved_image_paths_for_item:
            user_content_list.append({"type": "image"})

        # Construct the text part of the user message, matching training format
        text_for_user = f"弹幕时间序列: [{comment_seq_str}]\n\n弹幕文本序列: [{type_seq_str}]\n\n问题: {question_str}"
        user_content_list.append({"type": "text", "text": text_for_user})

        messages = [
            {"role": "system", "content": system_content_list}, # Note: Training script format puts system first
            {"role": "user", "content": user_content_list}
        ]
        all_messages_for_prompt.append(messages)

    if not all_messages_for_prompt:
        logger.error("Failed to generate any valid Task 5 test prompts. Check test data format and content.")
        sys.exit(1)

    logger.info(f"Loaded {len(all_messages_for_prompt)} Task 5 test prompts.")
    # Return messages, ground truths, and resolved image paths
    return all_messages_for_prompt, all_ground_truth_polarities, all_resolved_image_paths


def extract_polarity_from_response(response_text):
    """
    Extracts the predicted sentiment polarity value from the model's response text.
    Matches the format expected from train_task5.py's assistant response.
    Expected format: "时间窗口内的平均情感极性分数 {value}。"
    """
    # More specific regex based on the training output format
    match = re.search(r"时间窗口内的平均情感极性分数\s+(-?\d+\.?\d*)\s*。", response_text)
    if match:
        try:
            polarity = float(match.group(1))
            return polarity
        except ValueError:
            logger.warning(f"Extracted polarity value '{match.group(1)}' could not be converted to float from response: '{response_text}'")
            return None
    else:
        # Fallback: try to find any float if the exact pattern fails
        match_float = re.search(r"(-?\d+\.?\d+)", response_text)
        if match_float:
            try:
                polarity = float(match_float.group(1))
                logger.info(f"Used fallback float extraction: got {polarity} from '{response_text}'")
                return polarity
            except ValueError:
                 pass # Fall through
        logger.warning(f"Could not extract sentiment polarity using expected pattern from response: '{response_text}'")
        return None

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = Path(args.output_dir) / "test_task5_img_revised.log"
    # Configure logger to write to file
    logger.remove() # Remove default handler
    logger.add(sys.stderr, level="INFO") # Keep console output
    logger.add(log_file_path, level="INFO") # Add file handler
    logger.info(f"Revised Task 5 testing script started.")
    logger.info(f"Parsed arguments for Task 5 testing: {args}")

    logger.info(f"Loading base model processor from {args.base_model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        # Apply the same chat template used in training
        tokenizer.chat_template = USER_CHAT_TEMPLATE
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        sys.exit(1)

    # Handle pad_token and eos_token consistency with training script
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"pad_token was None, set to eos_token: {tokenizer.pad_token}")
        else:
            # Add a default pad token if neither exists (should match training logic if applicable)
            default_pad = "<|endoftext|>"
            tokenizer.add_special_tokens({'pad_token': default_pad})
            logger.warning(f"pad_token and eos_token were None. Added '{default_pad}' as pad_token.")
    if tokenizer.eos_token is None:
        # Qwen VL usually has <|im_end|> as eos, add it if missing
        default_eos = "<|im_end|>"
        tokenizer.add_special_tokens({'eos_token': default_eos})
        logger.warning(f"eos_token was None. Added '{default_eos}' as eos_token.")


    quantization_config = None
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.use_qlora:
        # Set up QLoRA config identical to training
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype, # Match compute dtype
            bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA is enabled. Loading base model with 4-bit quantization.")
        # When using QLoRA, model dtype is typically float16 or bfloat16 handled by BitsAndBytes
        model_dtype = None # Let BitsAndBytes handle the dtype

    logger.info(f"Loading base model '{args.base_model_path}'...")
    try:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_path,
            quantization_config=quantization_config,
            torch_dtype=model_dtype, # Set dtype only if not using QLoRA
            device_map="auto", # Use device mapping for potentially large models
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        sys.exit(1)

    # Resize embeddings if tokenizer added tokens and differs from model config logic check
    if len(tokenizer) > base_model.config.vocab_size:
        logger.info(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
        # If embeddings are resized after PeftModel.from_pretrained, it might cause issues.
        # It's generally better to ensure tokenizer and base model are aligned before loading PeftModel.


    logger.info(f"Loading PEFT model adapter from {args.peft_model_path}...")
    try:
        # Load the adapter onto the base model
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
        # Optional: Merge LoRA weights for faster inference if memory allows.
        # logger.info("Merging LoRA weights...")
        # model = model.merge_and_unload()
        # logger.info("LoRA weights merged.")
        logger.info("PEFT adapter loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load PEFT adapter from {args.peft_model_path}: {e}")
        logger.error("Ensure the path points to the directory containing 'adapter_config.json', 'adapter_model.safetensors', etc. (e.g., 'final_checkpoint').")
        sys.exit(1)

    model.eval() # Set the model to evaluation mode
    try:
        current_device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {current_device}")
    except Exception as e:
        logger.warning(f"Could not determine model device: {e}")
        current_device = "cuda" if torch.cuda.is_available() else "cpu" # Fallback device


    # Load data using the revised function
    all_messages, all_gt_polarities, all_image_paths_for_samples = load_test_data_task5(args.test_data_file, args)

    # Sanity checks
    if not all_messages or not all_gt_polarities:
        logger.error("No valid test data loaded. Exiting.")
        sys.exit(1)
    if len(all_messages) != len(all_gt_polarities) or len(all_messages) != len(all_image_paths_for_samples):
        logger.error(f"Data list length mismatch after loading: "
                     f"messages({len(all_messages)}), "
                     f"gt_polarities({len(all_gt_polarities)}), "
                     f"image_paths({len(all_image_paths_for_samples)}). Exiting.")
        sys.exit(1)


    predicted_polarities_parsed = []
    corresponding_ground_truths_for_metrics = []
    raw_model_responses = []
    failed_samples_indices = [] # Keep track of samples that failed processing/generation

    logger.info(f"Starting inference on {len(all_messages)} Task 5 test samples...")

    # Process samples batch by batch (current batch_size is 1)
    for i in tqdm(range(0, len(all_messages), args.batch_size), desc="Generating predictions for Task 5"):
        batch_indices = list(range(i, min(i + args.batch_size, len(all_messages))))
        batch_messages_structs = [all_messages[idx] for idx in batch_indices]
        batch_image_paths_lists = [all_image_paths_for_samples[idx] for idx in batch_indices]
        batch_gt_polarities = [all_gt_polarities[idx] for idx in batch_indices] # Keep GTs aligned

        batch_input_texts = []
        batch_pil_images_for_processor = [] # List of lists of PIL images, or list of Nones

        # Prepare inputs for the processor within the batch
        for k, sample_idx in enumerate(batch_indices): # k is index within batch, sample_idx is index in original list
            current_sample_messages = batch_messages_structs[k]
            current_sample_image_paths = batch_image_paths_lists[k]

            # 1. Apply chat template to get the text prompt
            # We need the 'assistant' prompt added to signal generation start
            try:
                text_prompt_from_template = tokenizer.apply_chat_template(
                    current_sample_messages,
                    tokenize=False,
                    add_generation_prompt=True # Crucial for inference
                )
                batch_input_texts.append(text_prompt_from_template)
            except Exception as e:
                 logger.error(f"Error applying chat template for sample index {sample_idx}: {e}. Skipping sample.")
                 failed_samples_indices.append(sample_idx)
                 # Need to add placeholders to keep batch alignment if skipping mid-batch
                 raw_model_responses.append("Error: Template application failed")
                 # Cannot proceed with this sample, but need to handle the rest of the batch if batch_size > 1
                 if args.batch_size > 1: batch_pil_images_for_processor.append(None) # Placeholder image list
                 continue # Skip to next item in batch

            # 2. Load PIL images for the current sample
            pil_images_for_current_sample = []
            if current_sample_image_paths:
                valid_images_found = False
                for img_path_idx, img_path in enumerate(current_sample_image_paths):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        # Optional: Resize here if needed, though processor should handle it.
                        # Match training resize logic if specified, e.g. mentions resize in preprocess
                        # target_size = (args.image_target_size_w, args.image_target_size_h)
                        # img = img.resize(target_size, Image.LANCZOS)
                        pil_images_for_current_sample.append(img)
                        valid_images_found = True
                    except FileNotFoundError:
                        logger.warning(f"Sample index {sample_idx}, Image index {img_path_idx}: File not found {img_path}.")
                    except Exception as e:
                        logger.warning(f"Sample index {sample_idx}, Image index {img_path_idx}: Failed to load image {img_path}: {e}.")
                # Append the list of images (or empty list) for this sample
                batch_pil_images_for_processor.append(pil_images_for_current_sample if valid_images_found else None)
            else:
                 # No images declared for this sample
                batch_pil_images_for_processor.append(None)

        # If all samples in the batch failed template application, skip processor/generation
        if len(batch_input_texts) == 0:
            logger.warning(f"Skipping batch starting at index {i} as no valid prompts could be generated.")
            continue

        # 3. Use processor for the batch
        try:
            inputs = processor(
                text=batch_input_texts,
                images=batch_pil_images_for_processor, # List of lists of PIL images or Nones
                return_tensors="pt",
                padding="longest", # Pad batch to longest sequence
                truncation="longest_first", # Truncate if needed
                max_length=args.max_seq_length # Use configured max length
            )
            # Move inputs to the model's device
            inputs = {k: v.to(current_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        except Exception as e:
            logger.error(f"Error during processor call for batch starting at index {i}: {e}. Skipping batch generation.")
            # Add error messages for all samples in this batch
            for sample_idx in batch_indices:
                if sample_idx not in failed_samples_indices: # Avoid double recording
                    failed_samples_indices.append(sample_idx)
                    raw_model_responses.append("Error: Processor failed")
            continue # Skip to next batch

        # 4. Generate responses
        try:
            with torch.no_grad():
                # Adjust max_new_tokens based on expected response length for Task 5
                # "时间窗口内的平均情感极性分数 -0.5。" is relatively short.
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=50, # Increased slightly for safety
                    eos_token_id=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else model.config.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.config.pad_token_id
                )

            # 5. Decode and process responses
            # Ensure we handle the case where generation output length differs from input batch size (if errors occurred)
            num_generated = generated_ids.shape[0]
            if num_generated != len(batch_input_texts):
                 logger.warning(f"Mismatch between input texts ({len(batch_input_texts)}) and generated sequences ({num_generated}) for batch starting at {i}. Trying to align.")
                 # Handle potential alignment issues if possible, otherwise log errors for affected samples

            # Decode each generated sequence in the batch
            for j in range(num_generated): # Iterate over generated sequences
                # Find corresponding original sample index and GT polarity
                # This assumes alignment holds, which might be wrong if errors occurred above
                original_sample_index = batch_indices[j] # This assumes the j-th generated output corresponds to the j-th valid input

                # Get input length to slice only the generated part
                # Need actual length *before* padding for the j-th item in the batch
                input_len = inputs['input_ids'][j].ne(tokenizer.pad_token_id).sum().item()
                response_ids = generated_ids[j][input_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                raw_model_responses.append(response_text) # Store raw response

                # Extract polarity and compare with ground truth
                gt_polarity_for_sample = batch_gt_polarities[j] # Get the corresponding GT
                pred_polarity = extract_polarity_from_response(response_text)

                if pred_polarity is not None:
                    predicted_polarities_parsed.append(pred_polarity)
                    corresponding_ground_truths_for_metrics.append(gt_polarity_for_sample)
                    logger.debug(f"Sample {original_sample_index}: GT={gt_polarity_for_sample:.4f}, Pred={pred_polarity:.4f}. Response: '{response_text}'")
                else:
                    logger.warning(f"Sample {original_sample_index}: Could not parse polarity. GT={gt_polarity_for_sample:.4f}. Response: '{response_text}'")
                    # Record failure for this specific sample if parsing fails
                    if original_sample_index not in failed_samples_indices:
                         failed_samples_indices.append(original_sample_index)

        except Exception as e:
             logger.error(f"Error during model generation or decoding for batch starting at index {i}: {e}")
             # Mark all potentially affected samples in the batch as failed
             for sample_idx in batch_indices:
                 if sample_idx not in failed_samples_indices:
                     failed_samples_indices.append(sample_idx)
                     # Add placeholder error message if not already added
                     if len(raw_model_responses) <= sample_idx: # Check index bounds
                         raw_model_responses.extend(["Error: Generation/Decoding failed"] * (sample_idx - len(raw_model_responses) + 1))
                     elif raw_model_responses[sample_idx] is None or "Error" not in raw_model_responses[sample_idx]:
                         raw_model_responses[sample_idx] = "Error: Generation/Decoding failed"


    # --- Metrics Calculation ---
    rmse = float('nan')
    mae = float('nan')
    num_total_samples = len(all_gt_polarities)
    num_successfully_parsed = len(predicted_polarities_parsed)
    num_failed_samples = len(set(failed_samples_indices)) # Unique failed indices
    num_metrics_calculated_on = len(corresponding_ground_truths_for_metrics)


    logger.info(f"Inference completed.")
    logger.info(f"Total samples: {num_total_samples}")
    logger.info(f"Samples with successfully parsed predictions: {num_successfully_parsed}")
    logger.info(f"Samples failed during processing/generation/parsing: {num_failed_samples}")
    logger.info(f"Number of prediction-GT pairs available for metrics: {num_metrics_calculated_on}")


    if num_metrics_calculated_on == 0:
        logger.error("No valid prediction-ground truth pairs available to calculate metrics.")
    elif num_successfully_parsed != num_metrics_calculated_on:
         logger.error(f"Mismatch between parsed predictions ({num_successfully_parsed}) and GTs for metrics ({num_metrics_calculated_on}). Skipping metrics calculation.")
    else:
        logger.info(f"Calculating metrics based on {num_metrics_calculated_on} successfully parsed prediction-GT pairs.")

        true_values = np.array(corresponding_ground_truths_for_metrics)
        pred_values = np.array(predicted_polarities_parsed)

        try:
            mse = mean_squared_error(true_values, pred_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_values, pred_values)

            logger.info(f"Sentiment Polarity Prediction RMSE: {rmse:.4f}")
            logger.info(f"Sentiment Polarity Prediction MAE: {mae:.4f}")
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")


    # --- Save Results ---
    # Summary file
    results_summary_path = Path(args.output_dir) / "test_summary_task5_img_revised.json"
    summary = {
        "base_model_path": args.base_model_path,
        "peft_model_path": args.peft_model_path,
        "test_data_file": args.test_data_file,
        "num_test_samples_total": num_total_samples,
        "num_valid_predictions_for_metrics": num_metrics_calculated_on,
        "num_failed_samples": num_failed_samples,
        "rmse": f"{rmse:.4f}" if not np.isnan(rmse) else "NaN",
        "mae": f"{mae:.4f}" if not np.isnan(mae) else "NaN",
    }
    try:
        with open(results_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        logger.info(f"Task 5 test summary saved to {results_summary_path}")
    except Exception as e:
        logger.error(f"Failed to save summary JSON: {e}")

    # Detailed predictions file (JSON Lines)
    detailed_results_path = Path(args.output_dir) / "detailed_predictions_task5_img_revised.jsonl"
    try:
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            # Ensure raw_model_responses has entries for all samples, even failures
            if len(raw_model_responses) < num_total_samples:
                 raw_model_responses.extend(["Error: Response not recorded"] * (num_total_samples - len(raw_model_responses)))

            for k in range(num_total_samples):
                # Determine status and prediction for logging
                status = "Success"
                parsed_pred_for_log = None
                raw_response = raw_model_responses[k]

                if k in failed_samples_indices or "Error:" in raw_response:
                     status = "Failed"
                else:
                    # Try parsing again for consistency in log, even if it failed before
                    parsed_pred_for_log = extract_polarity_from_response(raw_response)
                    if parsed_pred_for_log is None:
                        # Parsing failed here, even if not caught earlier as a hard failure
                        status = "Parsing Failed"


                result_line = {
                    "sample_index": k,
                    "status": status,
                    "raw_model_response": raw_response,
                    "predicted_polarity": parsed_pred_for_log, # Will be None if parsing failed
                    "ground_truth_polarity": all_gt_polarities[k],
                    "image_paths_for_sample": all_image_paths_for_samples[k],
                    # Optionally include the input prompt text (can be large)
                    # "input_prompt_text": tokenizer.apply_chat_template(all_messages[k], tokenize=False, add_generation_prompt=True) if k < len(all_messages) else "N/A"
                }
                f.write(json.dumps(result_line, ensure_ascii=False) + "\n")
        logger.info(f"Task 5 detailed predictions saved to {detailed_results_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed predictions JSONL: {e}")


    logger.info("Task 5 testing finished.")

if __name__ == "__main__":
    main()