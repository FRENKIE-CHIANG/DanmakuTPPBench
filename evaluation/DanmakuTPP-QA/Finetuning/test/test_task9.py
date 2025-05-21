# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from loguru import logger
from peft import PeftModel # 用于加载LoRA权重
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, # 确保这与您训练的模型一致
)
from tqdm.auto import tqdm
import evaluate # 引入evaluate库

# --- 和训练脚本中一致的常量 ---
USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# 训练时使用的系统提示 (从 train_task2_language.py 中确认)
# 训练脚本中 messages_for_prompt[0]["content"] 是 "你是一个有用的助手"
SYSTEM_PROMPT_FROM_TRAIN = "你是一个有用的助手"

#默认路径
DEFAULT_BASE_MODEL_PATH = "PATH_TO_YOUR_BASE_MODEL"
DEFAULT_ADAPTER_PATH = "PATH_TO_YOUR_ADAPTER_PATH"
DEFAULT_TEST_DATA_FILE = "PATH_TO_YOUR_TEST_DATA_FILE"
DEFAULT_OUTPUT_DIR = "PATH_TO_YOUR_OUTPUT_DIR"
#-------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Test a fine-tuned Qwen VL model for sentiment analysis report generation quality (BLEU, BERTScore).")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH,
                        help="Path to the original base Qwen VL model.")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH,
                        help="Path to the trained LoRA adapter checkpoint directory (e.g., final_checkpoint).")
    parser.add_argument("--dataset_file", type=str, default=DEFAULT_TEST_DATA_FILE,
                        help="Path to the JSON test data file for sentiment report task.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save detailed predictions, scores, and logs.")
    parser.add_argument("--max_raw_seq_len", type=int, default=100, # 与训练脚本的 max_raw_seq_len 一致
                        help="Maximum length of raw comment time and sentiment sequences. -1 for no truncation.")
    parser.add_argument("--max_seq_length", type=int, default=1750, # 与训练脚本的 max_seq_length 一致
                        help="Maximum tokenized sequence length for model input.")
    parser.add_argument("--batch_size", type=int, default=1, # 文本生成任务通常batch较小，可以根据显存调整
                        help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=512, # 情感报告可能较长，调整此参数
                        help="Max new tokens for the model to generate for the report.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu).")
    # BERTScore 相关参数
    parser.add_argument("--bertscore_lang", type=str, default="en",
                        help="Language for BERTScore model (e.g., 'en', 'zh'). Important for Chinese reports.")
    parser.add_argument("--bertscore_model_type", type=str, default=None, # 例如 "bert-base-chinese"
                        help="Specific model type for BERTScore. If None, uses default for the language.")

    args = parser.parse_args()
    return args

def load_model_and_processor(args):
    logger.info(f"Loading base model from {args.base_model_path}...")
    # 根据设备选择合适的torch_dtype
    torch_dtype_val = torch.bfloat16 if torch.cuda.is_bf16_supported() and args.device == "cuda" else torch.float32
    if args.device == "cpu": # CPU 不支持 bfloat16
        torch_dtype_val = torch.float32

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype_val,
        device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None, # 仅在CUDA可用时使用device_map
        trust_remote_code=True
    )
    if args.device == "cpu" and hasattr(base_model, 'to'):
        base_model.to(args.device)

    logger.info(f"Loading LoRA adapter from {args.adapter_path}...")
    if not os.path.exists(args.adapter_path) or not os.path.isdir(args.adapter_path):
        logger.error(f"Adapter path not found or is not a directory: {args.adapter_path}")
        logger.error("Please ensure --adapter_path is set correctly (e.g., to your 'final_checkpoint' directory).")
        sys.exit(1)
        
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = model.merge_and_unload() 
        logger.info("LoRA adapter merged and unloaded.")
    except Exception as e:
        logger.error(f"Failed to load or merge PEFT model from {args.adapter_path}: {e}")
        logger.error("Make sure the adapter_path points to a valid PEFT model directory (e.g., containing adapter_model.bin and adapter_config.json).")
        sys.exit(1)
        
    model.eval() 
    
    if args.device == "cuda" and torch.cuda.is_available() and not hasattr(model, 'hf_device_map'): # 如果device_map未生效
         model.to(args.device)
    elif args.device == "cpu": # 确保CPU模式
        model.to(args.device)


    logger.info(f"Loading processor...")
    # 优先从adapter_path加载，如果训练时保存了的话
    try:
        processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
        logger.info(f"Processor loaded from adapter_path: {args.adapter_path}")
    except Exception:
        logger.warning(f"Could not load processor from adapter_path '{args.adapter_path}', trying base_model_path '{args.base_model_path}'.")
        processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
        logger.info(f"Processor loaded from base_model_path: {args.base_model_path}")


    tokenizer = processor.tokenizer
    tokenizer.chat_template = USER_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"}) # Qwen common pad
        logger.info(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")

    if tokenizer.eos_token is None: # 确保eos_token存在
        tokenizer.add_special_tokens({'eos_token': "<|im_end|>"}) # Qwen common eos
        logger.info(f"Added eos_token: {tokenizer.eos_token}")
        
    return model, processor, tokenizer

def prepare_prompt_for_item(item, tokenizer, max_raw_seq_len_arg):
    # 与 train_task2_language.py 中的 load_and_prepare_dataset 一致
    original_comment_sequence = item.get("comment_time_sequence", [])
    if max_raw_seq_len_arg != -1 and len(original_comment_sequence) > max_raw_seq_len_arg:
        truncated_comment_sequence = original_comment_sequence[-max_raw_seq_len_arg:]
    else:
        truncated_comment_sequence = original_comment_sequence
    comment_seq_str = ", ".join(map(str, truncated_comment_sequence))

    original_sentiment_sequence = item.get("sentiment_sequence", [])
    if max_raw_seq_len_arg != -1 and len(original_sentiment_sequence) > max_raw_seq_len_arg:
        truncated_sentiment_sequence = original_sentiment_sequence[-max_raw_seq_len_arg:]
    else:
        truncated_sentiment_sequence = original_sentiment_sequence
    sentiment_seq_str = ", ".join(map(str, truncated_sentiment_sequence))

    # question_str 固定为训练时的样子
    question_str = "Please provide a detailed analysis of the sentiment trend in the bullet comments over time." 

    user_content = f"情感序列: [{sentiment_seq_str}]\n\n时间序列: [{comment_seq_str}]\n\n问题: {question_str}"

    messages_for_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_FROM_TRAIN}, # 使用训练时的系统提示
        {"role": "user", "content": user_content}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages_for_prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_text

def extract_report_from_generated_text(generated_text: str):
    """
    从模型生成的文本中提取情感分析报告。
    训练时的目标格式是 "sentiment_analysis_report: {report_content}。"
    """
    prefix = "sentiment_analysis_report:"
    if generated_text.startswith(prefix):
        report_content = generated_text[len(prefix):].strip()
        # 移除末尾的句号（如果有）
        if report_content.endswith("。"):
            report_content = report_content[:-1]
        return report_content
    # 如果前缀不匹配，可能模型没有按预期格式生成，直接返回原始文本（去除首尾空格）
    # 或者根据观察到的模型输出格式进行更复杂的解析
    logger.warning(f"Generated text did not start with '{prefix}'. Returning stripped text. Raw: '{generated_text}'")
    return generated_text.strip()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapter_name_part = Path(args.adapter_path).parent.name if Path(args.adapter_path).parent.name != "" else Path(args.adapter_path).name
    predictions_file = output_dir / f"report_predictions_{adapter_name_part}.jsonl"
    summary_file = output_dir / f"report_evaluation_summary_{adapter_name_part}.json"
    log_file = output_dir / f"test_report_eval_log_{adapter_name_part}.log"

    logger.remove() 
    logger.add(sys.stderr, level="INFO") 
    logger.add(log_file, level="INFO") 

    logger.info(f"Starting sentiment report generation test with arguments: {vars(args)}")
    logger.info(f"Detailed predictions will be saved to: {predictions_file}")
    logger.info(f"Evaluation summary will be saved to: {summary_file}")
    logger.info(f"Logs will be saved to: {log_file}")

    model, processor, tokenizer = load_model_and_processor(args)
    current_device = next(model.parameters()).device # 获取模型实际所在的设备
    logger.info(f"Model is currently on device: {current_device}")


    logger.info(f"Loading test dataset from {args.dataset_file}...")
    try:
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse test JSON file '{args.dataset_file}': {e}")
        sys.exit(1)

    if not isinstance(test_data, list):
        logger.error(f"Test dataset in '{args.dataset_file}' should be a list of dictionaries.")
        sys.exit(1)

    all_prompts = []
    all_ground_truth_reports = [] # 存储真实的报告文本
    all_original_items = [] 

    logger.info("Preparing prompts for sentiment report generation...")
    for item_idx, item in enumerate(tqdm(test_data, desc="Formatting prompts")):
        gt_report = item.get("sentiment_analysis_report") # 训练数据中的真实报告字段
        if gt_report is None or not isinstance(gt_report, str) or not gt_report.strip():
            video_id_info = item.get("video", item.get("id", f"item_index_{item_idx}"))
            logger.warning(f"Skipping item '{video_id_info}' due to missing or empty 'sentiment_analysis_report'. Found: '{gt_report}'")
            continue
        
        prompt = prepare_prompt_for_item(item, tokenizer, args.max_raw_seq_len)
        all_prompts.append(prompt)
        all_ground_truth_reports.append(gt_report.strip()) # 存储真实的报告
        all_original_items.append(item)

    if not all_prompts:
        logger.error("No valid prompts could be generated. Ensure test data contains 'sentiment_analysis_report'. Exiting.")
        sys.exit(1)
    
    total_valid_samples = len(all_prompts)
    logger.info(f"Total valid samples for report generation: {total_valid_samples}")

    all_generated_reports = []
    results_to_save = []

    logger.info(f"Starting inference on {total_valid_samples} samples with batch size {args.batch_size}...")

    for i in tqdm(range(0, total_valid_samples, args.batch_size), desc="Generating reports"):
        batch_prompts = all_prompts[i:i + args.batch_size]
        batch_gt_reports_slice = all_ground_truth_reports[i:i + args.batch_size]
        batch_original_items_slice = all_original_items[i:i + args.batch_size]

        inputs = processor(
            text=batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length # tokenizer的max_length
        )
        inputs = {k: v.to(current_device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": args.max_new_tokens, # 控制生成报告的长度
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": False, 
                # "num_beams": 3, # 可以尝试 beam search
                # "early_stopping": True,
            }
            if 'pixel_values' in inputs and inputs['pixel_values'] is None:
                 inputs.pop('pixel_values')

            generated_ids = model.generate(**generate_kwargs)
        
        batch_full_generated_texts = tokenizer.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:], # 只解码新生成的部分
            skip_special_tokens=True # 跳过特殊token，但可能保留如 <|im_end|> 这样的标记
        )

        for j, (full_gen_text, gt_report_text) in enumerate(zip(batch_full_generated_texts, batch_gt_reports_slice)):
            # 从模型输出中提取核心报告内容
            # 训练目标是 "sentiment_analysis_report: {report}。"
            # 因此，预测时也应该期待类似格式，或进行相应解析
            predicted_report_extracted = extract_report_from_generated_text(full_gen_text.strip())
            all_generated_reports.append(predicted_report_extracted)
            
            current_original_item = batch_original_items_slice[j]
            video_id_info = current_original_item.get("video", current_original_item.get("id", f"sample_index_{i+j}"))

            results_to_save.append({
                "item_identifier": video_id_info,
                # "prompt": batch_prompts[j],
                # "ground_truth_report": gt_report_text,
                "generated_text_raw_model_output": full_gen_text.strip(), # 模型直接输出
                "generated_report_extracted": predicted_report_extracted, # 解析后的报告
            })
            
            if len(results_to_save) % 20 == 0 or len(results_to_save) == total_valid_samples: # 减少日志频率
                logger.info(f"Processed {len(results_to_save)}/{total_valid_samples} samples for report generation...")

    # 保存所有预测结果到JSONL文件
    with open(predictions_file, 'w', encoding='utf-8') as outfile:
        for res_item in results_to_save:
            outfile.write(json.dumps(res_item, ensure_ascii=False) + "\n")
    logger.info(f"Detailed predictions and ground truths saved to: {predictions_file}")

    # --- 计算评估指标 ---
    bleu_score = None
    bert_score_results = None

    if total_valid_samples > 0 and len(all_generated_reports) == total_valid_samples:
        logger.info("Calculating BLEU score...")
        try:
            bleu_metric = evaluate.load('bleu')
            # BLEU 需要将真实参考变成列表的列表
            references_for_bleu = [[ref] for ref in all_ground_truth_reports]
            bleu_score = bleu_metric.compute(predictions=all_generated_reports, references=references_for_bleu)
            logger.info(f"BLEU Score: {bleu_score}")
        except Exception as e:
            logger.error(f"Failed to compute BLEU score: {e}")

        logger.info("Calculating BERTScore...")
        try:
            bertscore_metric = evaluate.load('bertscore')
            # BERTScore 需要指定语言，特别是对于中文
            # 对于中文，bert_score默认可能不是最优，可以尝试指定模型如 "bert-base-chinese"
            # bertscore_model_type = "bert-base-chinese" if args.bertscore_lang == "zh" else None
            bert_score_results = bertscore_metric.compute(
                predictions=all_generated_reports, 
                references=all_ground_truth_reports, 
                lang=args.bertscore_lang, # 例如 "zh" for Chinese
                model_type=args.bertscore_model_type # 可选，如 "bert-base-chinese"
            )
            # BERTScore 返回 P, R, F1 的列表，通常我们关心F1的平均值
            avg_f1 = sum(bert_score_results['f1']) / len(bert_score_results['f1']) if bert_score_results and 'f1' in bert_score_results and bert_score_results['f1'] else None
            logger.info(f"BERTScore (all results): {bert_score_results}")
            logger.info(f"BERTScore (Average F1): {avg_f1}")

        except Exception as e:
            logger.error(f"Failed to compute BERTScore: {e}")
            logger.error("Ensure you have 'bert_score' installed and a suitable model for the language (e.g., 'bert-base-chinese' for Chinese via --bertscore_model_type).")

    else:
        logger.warning("Not enough data for metric calculation or mismatch in generated/ground_truth counts.")

    # 保存评估摘要
    summary_data = {
        "adapter_path": args.adapter_path,
        "test_data_file": args.dataset_file,
        "total_valid_samples": total_valid_samples,
        "bleu_score": bleu_score,
        "bertscore_results_full": bert_score_results, # 保存完整的BERTScore字典
        "bertscore_avg_f1": avg_f1 if 'avg_f1' in locals() else None
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Evaluation summary saved to: {summary_file}")
    logger.info(f"Full log saved to: {log_file}")
    logger.info("Testing finished.")

if __name__ == "__main__":
    main()