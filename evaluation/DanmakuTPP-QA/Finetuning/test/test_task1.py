# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from loguru import logger
from peft import PeftModel 
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    GenerationConfig 
)
from tqdm.auto import tqdm
import re # 用于正则表达式解析


USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# --- 默认路径配置 ---
# 请根据您的实际环境和训练输出调整这些默认路径
DEFAULT_BASE_MODEL_PATH = "path to your base model"

DEFAULT_LORA_ADAPTER_PATH = "path to your LoRA adapter"
DEFAULT_TEST_DATASET_FILE = "path to your test dataset"
DEFAULT_OUTPUT_FILE = "path to your output file"
DEFAULT_MAX_RAW_COMMENT_SEQ_LEN = 80 
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_BATCH_SIZE = 4
# --- 结束默认路径配置 ---


def parse_args():
    parser = argparse.ArgumentParser(description="Test the accuracy of LoRA fine-tuned Qwen VL model for predicting danmaku peak counts.")
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH,
                        help="Path to the base pre-trained model.")
    parser.add_argument("--lora_adapter_path", type=str, default=DEFAULT_LORA_ADAPTER_PATH,
                        help="Path to the trained LoRA adapter weights.")
    parser.add_argument("--test_dataset_file", type=str, default=DEFAULT_TEST_DATASET_FILE,
                        help="Path to the test dataset JSON file.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="Path to save test results (including predictions and ground truth).")
    parser.add_argument("--max_raw_comment_seq_len", type=int, default=DEFAULT_MAX_RAW_COMMENT_SEQ_LEN,
                        help="Maximum length of raw comment time sequence for input prompt (number of entries). -1 means no truncation.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Maximum number of new tokens for model generation.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for inference.")
    args = parser.parse_args()
    return args

def extract_predicted_num_peaks(generated_text: str) -> int | None:
    """
    Extract predicted peak count from model generated text.
    Assumes model response format like "Peak count is X." or "The number of peaks is X."
    """
    match_cn = re.search(r"波峰数量为\s*(\d+)", generated_text)
    if match_cn:
        try:
            return int(match_cn.group(1))
        except ValueError:
            pass

    match_en = re.search(r"The number of peaks is\s*(\d+)", generated_text, re.IGNORECASE)
    if match_en:
        try:
            return int(match_en.group(1))
        except ValueError:
            pass
    
    numbers = re.findall(r'\d+', generated_text)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            pass
            
    logger.warning(f"Unable to extract peak count from generated text: '{generated_text}'")
    return None

def main():
    args = parse_args()
    # 确保输出文件的父目录存在
    output_file_path = Path(args.output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.remove() # 移除默认的 loguru handler
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
    logger.add(output_file_path.parent / f"test_log_{output_file_path.stem}.log", level="DEBUG") # 日志文件名与输出文件名关联

    logger.info(f"解析得到的参数（可能被默认值覆盖）: {args}")

    # --- 1. 加载模型和分词器/处理器 ---
    logger.info(f"从 {args.base_model_path} 加载基础模型和处理器...")
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    except Exception as e:
        logger.error(f"加载处理器失败于路径 {args.base_model_path}: {e}")
        logger.error("请确保基础模型路径正确并且包含有效的处理器文件。")
        sys.exit(1)
        
    tokenizer = processor.tokenizer
    tokenizer.chat_template = USER_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"tokenizer.pad_token 设置为 eos_token: {tokenizer.eos_token}")
        else:
            # 对于Qwen系列，<|endoftext|> 经常作为pad_token
            tokenizer.add_special_tokens({'pad_token': "<|endoftext|>"})
            logger.warning(f"tokenizer.pad_token 未设置且eos_token也为None，已添加 '<|endoftext|>' 作为 pad_token。")
    if tokenizer.eos_token is None:
        # 对于Qwen系列, <|im_end|> 是一个重要的特殊token，通常也用作对话轮次的结束
        tokenizer.add_special_tokens({'eos_token': "<|im_end|>"}) # 或者 tokenizer.model_eos_token
        logger.warning(f"tokenizer.eos_token 未设置，已添加 '<|im_end|>' 作为 eos_token。")
    
    tokenizer.padding_side = "left"

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"加载基础模型失败于路径 {args.base_model_path}: {e}")
        logger.error("请确保基础模型路径正确。")
        sys.exit(1)

    logger.info(f"从 {args.lora_adapter_path} 加载LoRA适配器权重...")
    if not Path(args.lora_adapter_path).exists() or not Path(args.lora_adapter_path, "adapter_model.safetensors").exists(): # 检查关键文件
        logger.error(f"LoRA适配器路径 {args.lora_adapter_path} 无效或不包含 'adapter_model.safetensors'。")
        logger.error("请提供正确的LoRA权重路径。如果模型未进行LoRA微调，请移除此步骤或使用不同的测试脚本。")
        # 可以选择是否退出，或者尝试不加载LoRA进行测试（如果逻辑允许）
        # sys.exit(1) 
        logger.warning("由于LoRA适配器路径无效，将尝试不加载LoRA权重进行测试。")
    else:
        try:
            model = PeftModel.from_pretrained(model, args.lora_adapter_path)
            model = model.merge_and_unload() 
            logger.info("LoRA权重已加载并合并。")
        except Exception as e:
            logger.error(f"加载或合并LoRA适配器失败: {e}")
            logger.warning("将尝试不使用LoRA权重进行测试。")


    model.eval()

    # --- 2. 加载测试数据集 ---
    logger.info(f"从 {args.test_dataset_file} 加载测试数据集...")
    if not Path(args.test_dataset_file).exists():
        logger.error(f"测试数据集文件 {args.test_dataset_file} 未找到。")
        sys.exit(1)
    try:
        with open(args.test_dataset_file, 'r', encoding='utf-8') as f:
            test_raw_data = json.load(f)
    except Exception as e:
        logger.error(f"加载或解析测试JSON文件失败: {e}")
        sys.exit(1)

    if not isinstance(test_raw_data, list) or not test_raw_data:
        logger.error("测试数据集应为一个非空字典列表。")
        sys.exit(1)

    # --- 3. 准备输入并进行模型推理 ---
    results = []
    correct_predictions = 0
    total_samples = len(test_raw_data)
    
    prompts_to_generate = []
    ground_truth_num_peaks = []
    original_indices = [] # 存储原始样本索引，以便与results对应

    MAX_COMMENT_SEQUENCE_LENGTH = args.max_raw_comment_seq_len

    for i, item in enumerate(test_raw_data):
        original_comment_sequence = item.get("comment_time_sequence", [])
        
        if MAX_COMMENT_SEQUENCE_LENGTH != -1 and len(original_comment_sequence) > MAX_COMMENT_SEQUENCE_LENGTH:
            truncated_comment_sequence = original_comment_sequence[-MAX_COMMENT_SEQUENCE_LENGTH:]
        else:
            truncated_comment_sequence = original_comment_sequence
            
        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))
        question_str = item.get("question", "")
        
        user_content = f"弹幕时间序列: [{comment_seq_str}]\n\n问题: {question_str}"
        messages_for_prompt = [
            {"role": "system", "content": "你是一个有用的助手."},
            {"role": "user", "content": user_content}
        ]
        
        prompt_text = tokenizer.apply_chat_template(
            messages_for_prompt, tokenize=False, add_generation_prompt=True
        )
        prompts_to_generate.append(prompt_text)
        
        try:
            gt_num_peaks = int(str(item.get("num_peaks", "0"))) 
            ground_truth_num_peaks.append(gt_num_peaks)
        except ValueError:
            logger.warning(f"样本 {i} (video: {item.get('video', 'N/A')}) 的真实 num_peaks ('{item.get('num_peaks')}') 无法转换为整数，将标记为None。")
            ground_truth_num_peaks.append(None)
        original_indices.append(i)


    logger.info(f"准备了 {len(prompts_to_generate)} 个prompt进行推理。")

    for i in tqdm(range(0, len(prompts_to_generate), args.batch_size), desc="模型推理中"):
        batch_prompts = prompts_to_generate[i:i+args.batch_size]
        batch_gt_num_peaks = ground_truth_num_peaks[i:i+args.batch_size]
        batch_original_indices = original_indices[i:i+args.batch_size]


        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        for j in range(outputs.shape[0]):
            input_len = inputs['input_ids'][j].shape[0]
            generated_sequence = outputs[j, input_len:]
            generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            original_sample_index = batch_original_indices[j]
            predicted_num_peaks = extract_predicted_num_peaks(generated_text.strip())
            actual_num_peaks = batch_gt_num_peaks[j]

            is_correct = False
            if predicted_num_peaks is not None and actual_num_peaks is not None:
                if predicted_num_peaks == actual_num_peaks:
                    correct_predictions += 1
                    is_correct = True
            
            # 从原始数据中获取 video ID，如果存在的话
            video_id_info = test_raw_data[original_sample_index].get("video", f"sample_{original_sample_index}")

            results.append({
                "video_id_or_sample_index": video_id_info,
                "prompt": batch_prompts[j], # 保存实际输入的prompt
                "generated_response": generated_text.strip(),
                "predicted_num_peaks": predicted_num_peaks,
                "actual_num_peaks": actual_num_peaks,
                "is_correct_num_peaks": is_correct
            })
            
            if (sum(1 for _ in results) % 50 == 0 or sum(1 for _ in results) == total_samples) :
                logger.info(f"已处理 {sum(1 for _ in results)}/{total_samples} 个样本...")

    valid_samples_for_accuracy = sum(1 for r in results if r['actual_num_peaks'] is not None and r['predicted_num_peaks'] is not None)
    if valid_samples_for_accuracy > 0:
        accuracy = (correct_predictions / valid_samples_for_accuracy) * 100
        logger.info(f"波峰数量预测准确率: {accuracy:.2f}% ({correct_predictions}/{valid_samples_for_accuracy} 有效样本)")
    else:
        logger.info("没有有效的样本用于计算准确率。")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"详细测试结果已保存到: {output_file_path}")

if __name__ == "__main__":
    main()