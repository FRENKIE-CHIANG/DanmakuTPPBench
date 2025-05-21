# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, # Using the specified model class
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from tqdm.auto import tqdm

# User-provided chat template (remains unchanged)
USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Default model path, please modify according to your actual model,
# e.g., "Qwen/Qwen2.5-VL-Chat" or "Qwen/Qwen1.5-VL-Chat"
DEFAULT_MODEL_PATH = "PATH_TO_YOUR_BASE_MODEL"
DEFAULT_DATA_FILE = "PATH_TO_YOUR_DATA_FILE"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a large language model (based on Qwen VL) for a new task with sentiment sequence.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the base model on Hugging Face Hub or local path.")
    parser.add_argument("--dataset_file", type=str, default=DEFAULT_DATA_FILE, help="Path to the JSON data file (e.g., test.json).")
    parser.add_argument("--output_dir", type=str, default="./results_new_task_sft_vl_task7-new", help="Output directory for the trained model and logs.") # MODIFIED: Changed default output dir
    parser.add_argument("--run_name_suffix", type=str, default="_new_task_sft_vl", help="Suffix for the run name in the output directory.") # MODIFIED: Changed suffix
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=1600, help="Maximum sequence length after tokenization.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps.")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")
    parser.add_argument("--use_qlora", action='store_true', help="Enable QLoRA (4-bit quantization).")
    parser.add_argument("--max_raw_seq_len", type=int, default=60, # MODIFIED: Generalized name for sequence length
                        help="Maximum length of raw comment time and sentiment sequences (number of items) to input into the prompt. -1 for no truncation.")
    args = parser.parse_args()
    args.run_name = Path(args.model_path).name + args.run_name_suffix
    args.actual_output_dir = Path(args.output_dir) / args.run_name
    return args

def load_and_prepare_dataset(dataset_file_path, tokenizer, args):
    logger.info(f"Loading dataset from {dataset_file_path}...")
    try:
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse JSON file: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        logger.error("Dataset should be a list of dictionaries.")
        sys.exit(1)

    processed_data = {'prompt': [], 'response': [], 'original_comment_len': [], 'original_sentiment_len': [], 'original_danmu_len':[]}
    MAX_RAW_SEQUENCE_LENGTH = args.max_raw_seq_len # Use the generalized argument

    for item in tqdm(raw_data, desc="Building prompt and response"):
        # Process comment_time_sequence
        original_comment_sequence = item.get("comment_time_sequence", [])
        processed_data['original_comment_len'].append(len(original_comment_sequence))
        if MAX_RAW_SEQUENCE_LENGTH != -1 and len(original_comment_sequence) > MAX_RAW_SEQUENCE_LENGTH:
            truncated_comment_sequence = original_comment_sequence[-MAX_RAW_SEQUENCE_LENGTH:]
        else:
            truncated_comment_sequence = original_comment_sequence
        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))

        # Process sentiment_sequence (NEW)
        original_sentiment_sequence = item.get("sentiment_sequence", [])
        processed_data['original_sentiment_len'].append(len(original_sentiment_sequence))
        if MAX_RAW_SEQUENCE_LENGTH != -1 and len(original_sentiment_sequence) > MAX_RAW_SEQUENCE_LENGTH:
            # Assuming sentiment_sequence and comment_time_sequence are parallel and should be truncated similarly
            truncated_sentiment_sequence = original_sentiment_sequence[-MAX_RAW_SEQUENCE_LENGTH:]
        else:
            truncated_sentiment_sequence = original_sentiment_sequence
        sentiment_seq_str = ", ".join(map(str, truncated_sentiment_sequence)) # Convert list of floats/ints to string

        question_str = item.get("question", "")

        # Process danmu_sequence
        # 弹幕本来就是str
        original_danmu_sequence = item.get("comment_sequence", [])
        processed_data['original_danmu_len'].append(len(original_danmu_sequence))
        if MAX_RAW_SEQUENCE_LENGTH != -1 and len(original_danmu_sequence) > MAX_RAW_SEQUENCE_LENGTH:
            # Assuming sentiment_sequence and comment_time_sequence are parallel and should be truncated similarly
            truncated_danmu_sequence = original_danmu_sequence[-MAX_RAW_SEQUENCE_LENGTH:]
        else:
            truncated_danmu_sequence = original_danmu_sequence
        danmu_seq_str = ", ".join(map(str, truncated_danmu_sequence)) # Convert list of floats/ints to string
        
        next_peak_time=item.get("next_peak_time", "")
        question_str = item.get("question", "")
        
        # MODIFIED: Construct user_content with sentiment_sequence, comment_time_sequence, and question
        user_content = f"时间序列: [{comment_seq_str}]\n\n弹幕文本序列: [{danmu_seq_str}]\n\n下一个波峰:{next_peak_time}\n\n问题: {question_str}"
        
        # MODIFIED: Use ground_truth for assistant_content
        assistant_content = f"弹幕情感极性分数 {str(item.get("ground_truth", ""))}。" # Ensure it's a string

        messages_for_prompt = [
            {"role": "system", "content": "你是一个有用的助手"}, # MODIFIED: System prompt can be adjusted if needed
            {"role": "user", "content": user_content}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages_for_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        processed_data['prompt'].append(prompt_text)
        processed_data['response'].append(assistant_content)

    if len(processed_data['prompt']) == 0:
        logger.error("No samples could be extracted from the data file. Please check the data format.")
        sys.exit(1)
    
    dataset = Dataset.from_dict(processed_data)
    
    all_train_dataset = DatasetDict({'train': dataset})
    logger.info(f"All data ({len(dataset)}) will be used for training.")
    
    logger.info(f"Training set size: {len(all_train_dataset['train'])}")
    
    return all_train_dataset

def preprocess_function(examples, tokenizer, max_seq_length):
    inputs_ids_list = []
    labels_list = []

    for prompt, response in zip(examples['prompt'], examples['response']):
        text_all = prompt + response + tokenizer.eos_token

        tokenized_all = tokenizer(
            text_all,
            max_length=max_seq_length,
            truncation=True,
            padding=False, # Padding will be handled by DataCollator
            return_tensors=None 
        )

        tokenized_prompt = tokenizer(
            prompt,
            max_length=max_seq_length, # Important to truncate prompt if it's too long
            truncation=True,
            padding=False,
            return_tensors=None
        )
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # Ensure prompt_len does not exceed the length of tokenized_all["input_ids"]
        # This can happen if the prompt alone is already longer than max_seq_length
        # or if prompt + response is shorter than prompt after truncation.
        # However, the logic should be: prompt tokens are part of input_ids, response tokens follow.
        # If text_all was truncated, prompt_len might be greater than actual len(input_ids)
        # if prompt itself was longer than max_seq_length.
        # The labels should mask out the prompt part of the `tokenized_all["input_ids"]`.
        
        current_labels = tokenized_all["input_ids"].copy()
        
        # Mask prompt tokens.
        # The number of prompt tokens to mask is min(prompt_len, len(tokenized_all["input_ids"]))
        # to avoid error if prompt itself got truncated to be shorter than its original tokenized length
        # but tokenized_all still contains some part of it.
        # More robustly, it's the part of tokenized_all["input_ids"] that corresponds to the prompt.
        
        # If the prompt itself was truncated to max_seq_length, then all input_ids are prompt.
        if prompt_len >= len(tokenized_all["input_ids"]):
            current_labels[:] = [-100] * len(tokenized_all["input_ids"])
        else:
            current_labels[:prompt_len] = [-100] * prompt_len
            
        # Ensure that if the combined length (prompt + response + eos) was truncated,
        # the labels for the truncated part of the response are also valid (not -100).
        # The current_labels already reflect tokenized_all, so if tokenized_all was truncated,
        # current_labels will also be truncated. The parts that are response should have valid labels.

        inputs_ids_list.append(tokenized_all["input_ids"])
        labels_list.append(current_labels)

    model_inputs = {
        "input_ids": inputs_ids_list,
        "labels": labels_list,
    }
    return model_inputs

def main():
    args = parse_args()
    os.makedirs(args.actual_output_dir, exist_ok=True)
    logger.add(args.actual_output_dir / "train.log")
    logger.info(f"Parsed arguments: {args}")
    logger.info(f"Output directory: {args.actual_output_dir}")

    logger.info(f"Loading processor and tokenizer from {args.model_path}...")
    # For Qwen VL models, AutoProcessor is typically used.
    # If your specific model requires a different processor class, adjust accordingly.
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer 
    tokenizer.chat_template = USER_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"Tokenizer's pad_token is not set. Using eos_token ({tokenizer.eos_token}) as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a common pad token if neither eos_token nor pad_token is set.
            default_pad_token = "<|endoftext|>" # Or another suitable token for Qwen VL
            tokenizer.add_special_tokens({'pad_token': default_pad_token})
            logger.warning(f"Neither eos_token nor pad_token was found. Added '{default_pad_token}' as pad_token. Please verify model compatibility.")

    # Ensure EOS token is set, crucial for generation.
    if tokenizer.eos_token is None:
        default_eos_token = "<|im_end|>" # A common EOS for Qwen models, verify for Qwen-VL
        tokenizer.add_special_tokens({'eos_token': default_eos_token})
        logger.warning(f"Tokenizer's eos_token is not set. Added '{default_eos_token}' as eos_token.")


    full_dataset = load_and_prepare_dataset(args.dataset_file, tokenizer, args)

    logger.info("Preprocessing dataset...")
    tokenized_datasets = DatasetDict()
    
    if 'train' in full_dataset:
        dataset_split = full_dataset['train']
        tokenized_datasets['train'] = dataset_split.map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_seq_length),
            batched=True,
            batch_size=1000, # Adjust batch size for mapping based on your system's memory
            num_proc=min(4, os.cpu_count() or 1), # Adjust num_proc based on your CPU cores
            remove_columns=[col for col in dataset_split.column_names if col not in ['input_ids', 'labels']]
        )
    else:
        logger.error("No 'train' split found in the loaded dataset. Exiting.")
        sys.exit(1)

    logger.info(f"Tokenized training set features: {tokenized_datasets['train'].features}")

    logger.info("--- Starting debug of preprocessed samples (first sample from training set) ---")
    if len(full_dataset['train']) > 0 and len(tokenized_datasets['train']) > 0:
        raw_sample_idx = 113
        
        raw_prompt_text_example = full_dataset['train'][raw_sample_idx]['prompt']
        raw_response_text_example = full_dataset['train'][raw_sample_idx]['response']
        original_comment_len_example = full_dataset['train'][raw_sample_idx].get('original_comment_len', 'N/A')
        original_sentiment_len_example = full_dataset['train'][raw_sample_idx].get('original_sentiment_len', 'N/A')


        logger.info(f"Sample {raw_sample_idx} - Original comment_time_sequence length (before truncation): {original_comment_len_example}")
        logger.info(f"Sample {raw_sample_idx} - Original sentiment_sequence length (before truncation): {original_sentiment_len_example}") # NEW
        logger.info(f"Sample {raw_sample_idx} - Text used for prompt (partial): {raw_prompt_text_example[:500]}...") # Show more for complex prompts
        logger.info(f"Sample {raw_sample_idx} - Text used for response: {raw_response_text_example}")

        text_all_debug = raw_prompt_text_example + raw_response_text_example + tokenizer.eos_token
        tokenized_all_debug = tokenizer(
            text_all_debug, max_length=args.max_seq_length, truncation=True, padding=False
        )
        input_ids_debug = tokenized_all_debug["input_ids"]
        final_tokenized_length = len(input_ids_debug)
        logger.info(f"Sample {raw_sample_idx} - 'text_all' (prompt+response+eos) tokenized and truncated to {args.max_seq_length}, final length: {final_tokenized_length}")

        tokenized_prompt_debug = tokenizer(
            raw_prompt_text_example, max_length=args.max_seq_length, truncation=True, padding=False
        )
        prompt_len_debug = len(tokenized_prompt_debug["input_ids"])
        logger.info(f"Sample {raw_sample_idx} - 'prompt_text' alone tokenized and truncated to {args.max_seq_length}, length (prompt_len): {prompt_len_debug}")
        
        tokenized_response_debug_only = tokenizer(raw_response_text_example, add_special_tokens=False) # Exclude special tokens for pure response length
        response_tokens_len_approx = len(tokenized_response_debug_only["input_ids"])
        logger.info(f"Sample {raw_sample_idx} - 'response_text' alone tokenized, approx. length (no eos): {response_tokens_len_approx}")


        processed_sample_labels = tokenized_datasets['train'][raw_sample_idx]['labels']
        valid_label_count = sum(1 for lbl in processed_sample_labels if lbl != -100)
        logger.info(f"Sample {raw_sample_idx} - Final Labels length: {len(processed_sample_labels)}")
        logger.info(f"Sample {raw_sample_idx} - Final Labels valid target tokens (non -100): {valid_label_count}")

        if prompt_len_debug >= final_tokenized_length and final_tokenized_length > 0 :
             logger.warning(f"Sample {raw_sample_idx} - WARNING: prompt_len ({prompt_len_debug}) >= final_tokenized_length ({final_tokenized_length})!")
             logger.warning(f"Sample {raw_sample_idx} - This means response part might be fully truncated or masked. Check max_seq_length and data.")
        elif final_tokenized_length <= prompt_len_debug :
             logger.warning(f"Sample {raw_sample_idx} - WARNING: final_tokenized_length ({final_tokenized_length}) <= prompt_len ({prompt_len_debug})!")
             logger.warning(f"Sample {raw_sample_idx} - This also means no space for response tokens to be learned.")


        if valid_label_count > 0:
            # logger.info(f"Sample {raw_sample_idx} - Labels (first 50): {processed_sample_labels[:50]}")
            # logger.info(f"Sample {raw_sample_idx} - Labels (last 50): {processed_sample_labels[-50:]}")
            valid_label_ids_for_decode = [l for l in processed_sample_labels if l != -100]
            decoded_valid_labels_text = tokenizer.decode(valid_label_ids_for_decode)
            logger.info(f"Sample {raw_sample_idx} - Decoded valid Labels (target text for model): '{decoded_valid_labels_text}'")
            # Compare with raw response
            if decoded_valid_labels_text.strip() != raw_response_text_example.strip() and not raw_response_text_example.strip().startswith(decoded_valid_labels_text.strip()): # Check if it's a prefix due to truncation
                 logger.warning(f"Sample {raw_sample_idx} - Decoded labels differ significantly from raw response. Raw: '{raw_response_text_example}', Decoded Labels: '{decoded_valid_labels_text}'. This might indicate truncation or issues.")
        else:
            logger.info(f"Sample {raw_sample_idx} - All labels are -100 or labels list is empty. No target for learning for this sample.")
            input_ids_from_processed = tokenized_datasets['train'][raw_sample_idx]['input_ids']
            decoded_input_from_processed = tokenizer.decode(input_ids_from_processed)
            logger.info(f"Sample {raw_sample_idx} - (For reference) Decoded actual input_ids for this sample: '{decoded_input_from_processed}'")

    else:
        logger.warning("Cannot perform sample debug as training dataset is empty.")
    logger.info("--- End of preprocessed sample debug ---")


    logger.info(f"Loading base model from {args.model_path}...")
    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA enabled, using 4-bit quantization.")

    # Ensure the model class is appropriate for your Qwen VL model version
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not args.use_qlora and torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available and not QLoRA
        device_map="auto", trust_remote_code=True
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Set use_gradient_checkpointing for QLoRA
        logger.info("Model prepared for k-bit training (QLoRA).")

    # LoRA Target Modules: These might need adjustment based on the specific Qwen-VL model architecture.
    # The example below is generic for many Qwen-like models.
    # You can print model.named_modules() to find correct module names for your specific model.
    example_target_modules = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'): # Common for newer HF Qwen models
        num_layers = len(model.model.layers)
        for i in range(num_layers):
            example_target_modules.extend([
                f"model.layers.{i}.self_attn.q_proj", f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj", f"model.layers.{i}.self_attn.o_proj",
                f"model.layers.{i}.mlp.gate_proj", f"model.layers.{i}.mlp.up_proj",
                f"model.layers.{i}.mlp.down_proj",
            ])
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): # Common for older/other HF Qwen models
         num_layers = len(model.transformer.h)
         for i in range(num_layers):
            example_target_modules.extend([
                f"transformer.h.{i}.attn.c_attn", f"transformer.h.{i}.attn.c_proj",
                # For some Qwen1.5 VL, MLP might be w1, w2 or gate_proj, up_proj, down_proj
                f"transformer.h.{i}.mlp.w1", # Or gate_proj
                f"transformer.h.{i}.mlp.w2", # Or up_proj
                f"transformer.h.{i}.mlp.c_proj", # Or down_proj
            ])
    # Add vision tower modules if you intend to fine-tune them (less common for LoRA on VLMs initially)
    # Example: "model.vision_tower.vision_tower.encoder.layers..."
    # For Qwen2.5-VL, it might be like: "model.vision_tower.vision_model.encoder.layers..."
    # It's often better to start without fine-tuning vision tower with LoRA unless specifically needed and resourced.

    if not example_target_modules:
        logger.warning("Could not automatically detect LoRA target modules. Using a generic fallback list. VERIFY AND UPDATE target_modules!")
        example_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        # It's crucial to ensure these target_modules are correct for Qwen2.5-VL
        target_modules=example_target_modules 
    )
    logger.info(f"LoRA target modules (example, VERIFY for your specific Qwen-VL model): {example_target_modules[:3]}... etc.")


    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"Failed to apply PEFT LoRA config: {e}")
        logger.error("Please double-check `lora_target_modules` for your model architecture.")
        logger.error("You can inspect module names by printing `model.named_modules()`.")
        sys.exit(1)

    training_args = TrainingArguments(
        output_dir=str(args.actual_output_dir), run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, logging_steps=args.logging_steps,
        save_steps=args.save_steps, 
        fp16=not (args.use_qlora or torch.cuda.is_bf16_supported()), # Enable fp16 if not using QLoRA and bf16 is not supported
        bf16=torch.cuda.is_bf16_supported() and not args.use_qlora, # Enable bf16 if supported and not using QLoRA
        eval_strategy="no", # Changed from "steps" to "no" as per original script logic
        # eval_steps=args.save_steps if args.train_on_dev and full_dataset.get('dev') else None, # No validation set
        save_total_limit=3,
        load_best_model_at_end=False, # Changed from True as no evaluation is performed
        report_to="tensorboard", remove_unused_columns=False, label_names=["labels"],
        gradient_checkpointing=args.use_qlora, # Enable if using QLoRA for memory saving
        # deepspeed=args.deepspeed_config if args.deepspeed_config else None, # If you plan to use DeepSpeed
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=None, # No validation set
        data_collator=data_collator, tokenizer=tokenizer
    )
    
    train_dataset_obj = tokenized_datasets.get("train")
    if not train_dataset_obj or len(train_dataset_obj) == 0:
         logger.error("No training data available. Exiting program."); sys.exit(1)

    logger.info("Starting training...")
    try:
        logger.info(f"Training on {len(train_dataset_obj)} samples.")
        
        train_result = trainer.train()

        logger.info("Saving final model (LoRA adapter weights)...")
        final_model_dir_str = str(args.actual_output_dir / "final_checkpoint")
        trainer.save_model(final_model_dir_str) 
        # Save processor along with the model for easy loading later
        processor.save_pretrained(final_model_dir_str)
        logger.info(f"Processor and final LoRA adapter weights saved to {final_model_dir_str}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.exception("An error occurred during training.")
        raise
    logger.info("Training complete.")

if __name__ == "__main__":
    main()