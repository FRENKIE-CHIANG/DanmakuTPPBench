# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import torch
import numpy as np 
from pathlib import Path
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, 
    BitsAndBytesConfig
)
from tqdm import tqdm
from PIL import Image


# 与训练脚本一致的聊天模板
USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Task 13 默认路径
DEFAULT_BASE_MODEL_PATH = "PATH_TO_YOUR_BASE_MODEL"
DEFAULT_PEFT_MODEL_PATH_TASK13 = "PATH_TO_YOUR_PEFT_MODEL_PATH"
DEFAULT_TEST_DATA_FILE_TASK13 = "PATH_TO_YOUR_TEST_DATA_FILE"
DEFAULT_OUTPUT_DIR_TASK13 = "PATH_TO_YOUR_OUTPUT_DIR"
DEFAULT_IMAGE_BASE_PATH = "PATH_TO_YOUR_IMAGE_BASE_PATH"

# Task 13 中训练时使用的固定前缀和后缀 (来自 assistant_content 格式)
TRAINING_DATA_PREFIX = "弹幕爆发波峰的原因分析报告 "
TRAINING_DATA_SUFFIX = "。"

def parse_args():
    parser = argparse.ArgumentParser(description="为 Task 13 (弹幕波峰原因报告生成) 测试微调后的 Qwen VL 模型，并清理固定格式。")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH, help="基础 Hugging Face 模型路径。")
    parser.add_argument("--peft_model_path", type=str, default=DEFAULT_PEFT_MODEL_PATH_TASK13, help="训练好的 Task 13 PEFT (LoRA) 适配器路径。")
    parser.add_argument("--test_data_file", type=str, default=DEFAULT_TEST_DATA_FILE_TASK13, help="Task 13 JSON 测试数据文件路径。")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR_TASK13, help="保存 Task 13 测试结果和日志的目录。")
    parser.add_argument("--image_base_path", type=str, default=DEFAULT_IMAGE_BASE_PATH, help="JSON 中相对图像路径的基本路径。")
    parser.add_argument("--max_seq_length", type=int, default=8182, help="分词的最大序列长度。")
    parser.add_argument("--max_raw_seq_len", type=int, default=100, help="prompt 中原始序列的最大长度。")
    parser.add_argument("--max_report_length", type=int, default=2000, help="生成报告的最大 token 数量。")
    parser.add_argument("--batch_size", type=int, default=1, help="推理的批量大小。")
    parser.add_argument("--use_qlora", action='store_true', help="如果基础模型在训练期间使用 QLoRA 加载，则设置此项。")
    # strip_known_format 参数现在控制是否移除前缀和后缀
    parser.add_argument("--strip_known_format", action='store_true', default=True, help="是否从生成的报告中移除已知的固定前缀和后缀(句号)。")
    return parser.parse_args()

def load_test_data_task13(data_file_path, args):
    """为 Task 13 加载测试数据并准备 prompt，与 train_task13_text.py 一致。"""
    logger.info(f"为 Task 13 从 {data_file_path} 加载测试数据...")
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"加载或解析 JSON 文件失败: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        logger.error("测试数据应为字典列表。")
        sys.exit(1)

    all_messages_for_prompt = []
    all_ground_truth_reports = []
    all_resolved_image_paths = []
    skipped_samples_count = 0

    for item_idx, item in enumerate(tqdm(raw_data, desc="准备 Task 13 测试 prompt")):
        gt_report_raw = item.get("ground_truth") # 这是训练时 {} 中的内容
        if gt_report_raw is None or not isinstance(gt_report_raw, str):
            logger.warning(f"条目 {item_idx} (ID: {item.get('id', 'N/A')}) 'ground_truth' 缺失或非字符串。跳过。")
            skipped_samples_count += 1
            continue
        # 对于 Task 13，ground_truth 就是核心报告内容，不需要再包装前缀后缀
        all_ground_truth_reports.append(str(gt_report_raw))


        original_comment_sequence = item.get("comment_time_sequence", [])
        if args.max_raw_seq_len != -1 and len(original_comment_sequence) > args.max_raw_seq_len:
            truncated_comment_sequence = original_comment_sequence[-args.max_raw_seq_len:]
        else:
            truncated_comment_sequence = original_comment_sequence
        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))

        original_type_sequence = item.get("comment_sequence", [])
        if args.max_raw_seq_len != -1 and len(original_type_sequence) > args.max_raw_seq_len:
            truncated_type_sequence = original_type_sequence[-args.max_raw_seq_len:]
        else:
            truncated_type_sequence = original_type_sequence
        type_seq_str = ", ".join(map(str, truncated_type_sequence))

        event_type_proportion = item.get("event_type_proportion", "")
        object_tags_in_peak_frame = item.get("object_tags_in_peak_frame", "")
        peak_sentiment = item.get("peak_sentiment", "")
        question_str = item.get("question", "")

        image_paths_relative = item.get("video_frames_in_peak_time_window", [])
        resolved_image_paths_for_item = []
        image_base_dir = Path(args.image_base_path)
        if image_paths_relative:
            for rel_path in image_paths_relative:
                full_path = Path(rel_path)
                if args.image_base_path and not full_path.is_absolute():
                    resolved_image_paths_for_item.append(str(image_base_dir / rel_path))
                else:
                    resolved_image_paths_for_item.append(str(full_path))
        all_resolved_image_paths.append(resolved_image_paths_for_item)

        system_content_list = [{"type": "text", "text": "你是一个有用的助手"}]
        user_content_list = []
        for _ in resolved_image_paths_for_item:
            user_content_list.append({"type": "image"})

        text_for_user = (
            f"弹幕时间序列: [{comment_seq_str}]\n\n"
            f"弹幕文本序列: [{type_seq_str}]\n\n"
            f"事件类型情况: {event_type_proportion}\n\n"
            f"ram标签: {object_tags_in_peak_frame}\n\n"
            f"波峰情感极性分数: {peak_sentiment}\n\n"
            f"问题: {question_str}"
        )
        user_content_list.append({"type": "text", "text": text_for_user})

        messages = [
            {"role": "system", "content": system_content_list},
            {"role": "user", "content": user_content_list}
        ]
        all_messages_for_prompt.append(messages)

    if skipped_samples_count > 0:
        logger.info(f"因 'ground_truth' 无效或缺失，跳过了 {skipped_samples_count} 个样本。")
    if not all_messages_for_prompt:
        logger.error("未能生成任何有效的 Task 13 测试 prompt。检查测试数据格式和内容。")
        sys.exit(1)

    logger.info(f"加载了 {len(all_messages_for_prompt)} 个 Task 13 测试 prompt。")
    return all_messages_for_prompt, all_ground_truth_reports, all_resolved_image_paths

def clean_generated_report(raw_text, strip_format_flag):
    """根据训练模板清理生成的报告文本，移除固定的前缀和后缀（句号）。"""
    if not strip_format_flag:
        return raw_text

    text_after_prefix = raw_text
    if raw_text.startswith(TRAINING_DATA_PREFIX):
        text_after_prefix = raw_text[len(TRAINING_DATA_PREFIX):]

    # 移除末尾的句号，这个句号是训练模板在 ground_truth 之后添加的
    cleaned_text = text_after_prefix
    if text_after_prefix.endswith(TRAINING_DATA_SUFFIX):
        # 确保不是意外移除了 ground_truth 本身就有的、且恰好也是句号的结尾
        # 通过比较原始文本是否严格符合 "前缀 + (核心内容不带句号) + 句号" 的模式
        # 简化处理：如果移除了前缀后，文本以训练模板的后缀结尾，则移除该后缀
        cleaned_text = text_after_prefix[:-len(TRAINING_DATA_SUFFIX)]

    return cleaned_text.strip()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = Path(args.output_dir) / "test_task13_report_cleaned.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file_path, level="INFO")

    logger.info("Task 13 (报告生成与清理) 测试脚本已启动。")
    logger.info(f"解析的参数: {args}")
    if args.strip_known_format:
        logger.info(f"将尝试移除前缀: '{TRAINING_DATA_PREFIX}' 和后缀: '{TRAINING_DATA_SUFFIX}'")

    logger.info(f"从 {args.base_model_path} 加载基础模型处理器...")
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        tokenizer.chat_template = USER_CHAT_TEMPLATE
    except Exception as e:
        logger.error(f"加载处理器失败: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': "<|im_end|>"})

    quantization_config = None
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=model_dtype, bnb_4bit_use_double_quant=True)
        model_dtype = None

    logger.info(f"加载基础模型 '{args.base_model_path}'...")
    try:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_path, quantization_config=quantization_config, torch_dtype=model_dtype,
            device_map="auto", trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"加载基础模型失败: {e}")
        sys.exit(1)

    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    logger.info(f"从 {args.peft_model_path} 加载 PEFT 模型适配器...")
    try:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    except Exception as e:
        logger.error(f"加载 PEFT 适配器失败: {e}")
        sys.exit(1)

    model.eval()
    current_device = next(model.parameters()).device
    logger.info(f"模型已加载到设备: {current_device}")

    all_messages_structs, all_gt_reports, all_image_paths_for_samples = load_test_data_task13(args.test_data_file, args)

    if not all_messages_structs or not all_gt_reports or \
       len(all_messages_structs) != len(all_gt_reports) or \
       len(all_messages_structs) != len(all_image_paths_for_samples):
        logger.error("测试数据加载不一致或为空。正在退出。")
        sys.exit(1)

    generated_reports_texts = [""] * len(all_messages_structs)
    failed_samples_indices = []

    logger.info(f"在 {len(all_messages_structs)} 个 Task 13 测试样本上开始推理...")

    for i in tqdm(range(0, len(all_messages_structs), args.batch_size), desc="为 Task 13 生成报告"):
        batch_indices = list(range(i, min(i + args.batch_size, len(all_messages_structs))))
        # ... (batch_messages_structs, batch_image_paths_lists 定义同前) ...
        batch_messages_structs = [all_messages_structs[idx] for idx in batch_indices]
        batch_image_paths_lists = [all_image_paths_for_samples[idx] for idx in batch_indices]

        batch_input_texts = []
        batch_pil_images_for_processor = []

        for k, sample_idx in enumerate(batch_indices):
            current_sample_messages = batch_messages_structs[k]
            current_sample_image_paths = batch_image_paths_lists[k]

            try:
                text_prompt_from_template = tokenizer.apply_chat_template(
                    current_sample_messages, tokenize=False, add_generation_prompt=True
                )
                batch_input_texts.append(text_prompt_from_template)
            except Exception as e:
                logger.error(f"样本 {sample_idx}: 模板应用错误: {e}。")
                failed_samples_indices.append(sample_idx)
                generated_reports_texts[sample_idx] = "错误: 模板应用失败"
                if args.batch_size > 1: batch_pil_images_for_processor.append(None)
                continue

            pil_images_for_current_sample = []
            if current_sample_image_paths:
                valid_images_found = False
                for img_path in current_sample_image_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        pil_images_for_current_sample.append(img)
                        valid_images_found = True
                    except Exception as e:
                        logger.warning(f"样本 {sample_idx}: 加载图像 {img_path} 失败: {e}。")
                batch_pil_images_for_processor.append(pil_images_for_current_sample if valid_images_found else None)
            else:
                batch_pil_images_for_processor.append(None)

        if not batch_input_texts:
            continue

        try:
            inputs = processor(
                text=batch_input_texts, images=batch_pil_images_for_processor,
                return_tensors="pt", padding="longest", truncation="longest_first",
                max_length=args.max_seq_length
            )
            inputs = {k_val: v.to(current_device) for k_val, v in inputs.items() if isinstance(v, torch.Tensor)}
        except Exception as e:
            logger.error(f"批次 (起始索引 {i}): 处理器错误: {e}。")
            for abs_sample_idx in batch_indices:
                if abs_sample_idx not in failed_samples_indices:
                    failed_samples_indices.append(abs_sample_idx)
                    generated_reports_texts[abs_sample_idx] = "错误: 处理器失败"
            continue

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=args.max_report_length,
                    eos_token_id=[tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id
                )

            for j in range(generated_ids.shape[0]):
                original_sample_index = batch_indices[j]
                if original_sample_index in failed_samples_indices: continue

                try:
                    prompt_len = inputs['input_ids'][j].ne(tokenizer.pad_token_id).sum().item()
                    response_ids_only = generated_ids[j][prompt_len:]
                    response_text_raw = tokenizer.decode(response_ids_only, skip_special_tokens=True).strip()
                    
                    cleaned_report = clean_generated_report(response_text_raw, args.strip_known_format)
                    
                    generated_reports_texts[original_sample_index] = cleaned_report
                    logger.debug(f"样本 {original_sample_index}: 清理后报告 (片段): '{cleaned_report[:100]}...'")
                except Exception as decode_err:
                    logger.error(f"样本 {original_sample_index}: 解码错误: {decode_err}")
                    generated_reports_texts[original_sample_index] = "错误: 解码失败"
                    if original_sample_index not in failed_samples_indices:
                        failed_samples_indices.append(original_sample_index)
        except Exception as e:
            logger.error(f"批次 (起始索引 {i}): 生成/解码错误: {e}")
            for abs_sample_idx in batch_indices:
                if abs_sample_idx not in failed_samples_indices:
                    failed_samples_indices.append(abs_sample_idx)
                    generated_reports_texts[abs_sample_idx] = "错误: 生成/解码失败"

    num_total_samples = len(all_gt_reports)
    num_generation_failed = len(set(failed_samples_indices))

    logger.info(f"推理完成。总样本数: {num_total_samples}, 失败样本数: {num_generation_failed}")

    results_summary_path = Path(args.output_dir) / "test_summary_task13_report_cleaned.json"
    summary = {
        "base_model_path": args.base_model_path, "peft_model_path": args.peft_model_path,
        "test_data_file": args.test_data_file, "num_test_samples_total": num_total_samples,
        "num_failed_samples_during_generation": num_generation_failed,
        "max_report_length_setting": args.max_report_length,
        "known_format_stripped": args.strip_known_format
    }
    with open(results_summary_path, 'w', encoding='utf-8') as f: json.dump(summary, f, indent=4, ensure_ascii=False)
    logger.info(f"Task 13 测试摘要已保存到 {results_summary_path}")

    detailed_results_path = Path(args.output_dir) / "detailed_predictions_task13_report_cleaned.jsonl"
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        for k in range(len(all_messages_structs)): # 使用 all_messages_structs 的长度，因为它代表尝试处理的样本
            status = "成功" if k not in failed_samples_indices and "错误:" not in generated_reports_texts[k] else "失败"
            generated_report_text = generated_reports_texts[k]
            gt_report_text = all_gt_reports[k] if k < len(all_gt_reports) else "N/A"
            current_image_paths = all_image_paths_for_samples[k] if k < len(all_image_paths_for_samples) else []

            result_line = {
                "sample_index": k, "status": status,
                "generated_report_cleaned": generated_report_text, # 名称更改以反映清理
                "ground_truth_report": gt_report_text,
                "image_paths_for_sample": current_image_paths
            }
            f.write(json.dumps(result_line, ensure_ascii=False) + "\n")
    logger.info(f"Task 13 详细生成的报告已保存到 {detailed_results_path}")
    logger.info("Task 13 测试完成。")

if __name__ == "__main__":
    main()