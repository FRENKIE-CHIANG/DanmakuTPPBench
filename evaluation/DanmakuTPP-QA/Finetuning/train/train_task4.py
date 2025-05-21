# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path
import torch
from datasets import Dataset, DatasetDict # 确保 Dataset 被正确导入
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration, # 使用指定的模型类
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from tqdm.auto import tqdm
from PIL import Image

# 用户提供的聊天模板 (保持不变)
USER_CHAT_TEMPLATE = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

DEFAULT_MODEL_PATH = "PATH_TO_YOUR_BASE_MODEL"
DEFAULT_DATA_FILE = "PATH_TO_YOUR_DATA_FILE"

def parse_args():
    parser = argparse.ArgumentParser(description="使用情感序列和图像为新任务微调大型语言模型 (基于 Qwen VL)。")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Hugging Face Hub 上的基础模型路径或本地路径。")
    parser.add_argument("--dataset_file", type=str, default=DEFAULT_DATA_FILE, help="JSON 训练数据文件路径 (例如 train.json)。")
    parser.add_argument("--output_dir", type=str, default="./results_new_task_sft_vl_task5_img", help="训练模型和日志的输出目录。")
    parser.add_argument("--run_name_suffix", type=str, default="_new_task_sft_vl_img", help="输出目录中运行名称的后缀。")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个 GPU 的训练批量大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率。")
    parser.add_argument("--max_seq_length", type=int, default=8182, help="分词后的最大序列长度。注意：Qwen-VL 对文本和图像token可能有不同的限制")
    parser.add_argument("--logging_steps", type=int, default=10, help="每 X 步记录一次。")
    parser.add_argument("--save_steps", type=int, default=100, help="每 X 步保存一次检查点。")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r 参数。")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha 参数。")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout 率。")
    parser.add_argument("--use_qlora", action='store_true', help="启用 QLoRA (4位量化)。")
    parser.add_argument("--max_raw_seq_len", type=int, default=50,
                        help="输入到 prompt 中的原始评论时间和情感序列的最大长度 (项目数)。-1 表示不截断。")
    parser.add_argument("--image_base_path", type=str, default="/root/autodl-tmp/MMTPP", help="如果JSON中的路径是相对的，则为图像文件添加基本路径。")
    parser.add_argument("--skip_sample_if_all_images_missing", action='store_true', help="如果设置，当样本声明有图像但所有图像都找不到时，跳过该样本。")


    args = parser.parse_args()
    args.run_name = Path(args.model_path).name + args.run_name_suffix
    args.actual_output_dir = Path(args.output_dir) / args.run_name
    return args


def load_and_prepare_dataset(dataset_file_path, args): # tokenizer 不再需要在这里传递
    logger.info(f"从 {dataset_file_path} 加载数据集...")
    try:
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"加载或解析JSON文件失败: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        logger.error("数据集应该是一个字典列表。")
        sys.exit(1)

    # processed_data 的键现在直接对应 Dataset 的列名
    # 我们将直接构建一个字典列表，每个字典是一个样本
    all_samples_for_dataset = []
    skipped_data_count_initial = 0

    for item_idx, item in enumerate(tqdm(raw_data, desc="构建初始样本结构")):
        try:
            # 检查 ground_truth 是否存在 (对于训练至关重要)
            if item.get("ground_truth") is None:
                logger.warning(f"第 {item_idx} 条数据缺少 'ground_truth'，将跳过此条数据。")
                skipped_data_count_initial += 1
                continue

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

            question_str = item.get("question", "")
            image_paths_relative = item.get("video_frames_in_peak_time_window", [])

            system_content_list = [{"type": "text", "text": "你是一个有用的助手"}]
            user_content_list = []
            resolved_image_paths = []
            image_base_dir = Path(args.image_base_path)

            if image_paths_relative: # 只有当JSON中声明有图像时才处理
                for rel_path in image_paths_relative:
                    if args.image_base_path and not Path(rel_path).is_absolute():
                        resolved_image_paths.append(str(image_base_dir / rel_path))
                    else:
                        resolved_image_paths.append(rel_path)
            
            # 图像占位符现在只根据解析出的 resolved_image_paths 添加
            for _ in resolved_image_paths:
                user_content_list.append({"type": "image"})

            text_for_user = f"弹幕时间序列: [{comment_seq_str}]\n\n弹幕文本序列: [{type_seq_str}]\n\n问题: {question_str}"
            user_content_list.append({"type": "text", "text": text_for_user})

            messages = [
                {"role": "system", "content": system_content_list},
                {"role": "user", "content": user_content_list}
            ]
            
            assistant_content = f"时间窗口内的平均情感极性分数 {str(item.get("ground_truth", ""))}。"

            sample = {
                "messages": messages,
                "response": assistant_content,
                "image_files": resolved_image_paths, # 存储解析后的路径列表
                "original_comment_len": comment_len,
                "original_type_len": type_len,
                "id_in_json": item_idx # 保留原始索引以供调试
            }
            all_samples_for_dataset.append(sample)

        except Exception as e:
            logger.error(f"构建第 {item_idx} 条消息和响应时发生初始错误: {e}，将跳过此条数据。")
            skipped_data_count_initial += 1
            continue
    
    if skipped_data_count_initial > 0:
        logger.info(f"在 load_and_prepare_dataset 阶段，因数据格式或缺少ground_truth跳过了 {skipped_data_count_initial} 条数据。")

    if not all_samples_for_dataset:
        logger.error("未能从JSON文件中加载任何有效数据。")
        sys.exit(1)
    
    # 使用 Dataset.from_list (Hugging Face v2.17+ 推荐) 或 Dataset.from_dict
    # 为了兼容性，如果 from_list 不可用，则转换回 dict of lists
    try:
        dataset = Dataset.from_list(all_samples_for_dataset)
    except AttributeError: # 对于旧版 datasets
        logger.warning("Dataset.from_list 不可用，尝试 Dataset.from_dict...")
        if not all_samples_for_dataset: # 再次检查以防万一
             logger.error("转换到 Dataset.from_dict 前数据为空。"); sys.exit(1)
        # 将 list of dicts 转换为 dict of lists
        dict_of_lists = {key: [dic[key] for dic in all_samples_for_dataset] for key in all_samples_for_dataset[0]}
        dataset = Dataset.from_dict(dict_of_lists)

    logger.info(f"初步从JSON加载并结构化的有效数据条数: {len(dataset)}")
    return dataset


# 全局计数器，用于在 preprocess_function (如果以非batched方式或特殊方式调用时) 或外部统计
# 但由于 .map(batched=True) 的并行性，直接在 preprocess_function 中修改全局变量不可靠。
# 我们将通过 preprocess_function 返回一个有效性标记。

def preprocess_function(examples, processor, max_seq_length, tokenizer_for_labels, args):
    batch_size = len(examples['messages'])
    # 初始化 results，增加 image_grid_thw
    results = {
        "input_ids": [[] for _ in range(batch_size)],
        "attention_mask": [[] for _ in range(batch_size)],
        "labels": [[] for _ in range(batch_size)],
        "pixel_values": [None for _ in range(batch_size)],
        "image_grid_thw": [None for _ in range(batch_size)], # 新增
        # "is_valid_sample": [False for _ in range(batch_size)] # 移除
    }
    # 移除 skipped 计数器，因为不再基于 valid_sample 过滤
    # num_skipped_this_batch_filenotfound = 0
    # num_skipped_this_batch_indexerror = 0
    # num_skipped_this_batch_other_error = 0

    for i in range(batch_size):
        current_messages = examples['messages'][i]
        response_text = examples['response'][i]
        image_paths_for_sample = examples['image_files'][i]
        sample_id_for_log = examples.get('id_in_json', ['N/A']*batch_size)[i]

        try:
            pil_images = []
            num_images_loaded = 0
            if image_paths_for_sample:
                for img_path_idx, img_path in enumerate(image_paths_for_sample):
                    try:
                        # 直接打开图像，不进行 resize 或 convert (processor会处理)
                        img = Image.open(img_path)
                        # img.resize((64,64))
                        # 确保图像至少能被打开
                        # img.load() # 尝试加载数据，捕获潜在的解码错误
                        pil_images.append(img)
                        num_images_loaded += 1
                    # FileNotFoundError 理论上不应再发生，但保留以防万一
                    except FileNotFoundError:
                        logger.warning(f"样本 {sample_id_for_log}, 图像索引 {img_path_idx}: 找不到图像文件 {img_path}。")
                    except Exception as e: # 捕获包括解码在内的其他加载错误
                        logger.warning(f"样本 {sample_id_for_log}, 图像索引 {img_path_idx}: 加载或解码图像 {img_path} 失败: {e}。")

            # 如果没有成功加载任何图像，则 processor 的 images 参数为 None
            current_pil_images_for_processor = pil_images if pil_images else None

            text_prompt_from_template = tokenizer_for_labels.apply_chat_template(
                current_messages, tokenize=False, add_generation_prompt=True
            )

            # 调用 Processor
            try:
                 inputs = processor(
                    text=[text_prompt_from_template],
                    images=current_pil_images_for_processor, # 传递 PIL 图像列表或 None
                    return_tensors="pt",
                    truncation="longest_first",
                    max_length=max_seq_length,
                    padding=False # 不在这里 padding
                 )
            except Exception as e:
                 logger.error(f"样本 {sample_id_for_log}: 调用 Processor 时发生错误: {e}。跳过此样本的后续处理。")
                 # results 中的所有字段将保持默认值 (空列表或 None)
                 continue # 进行下一个样本

            # --- 文本部分处理 (保持不变) ---
            if inputs['input_ids'].ndim == 1:
                inputs['input_ids'] = inputs['input_ids'].unsqueeze(0)
                if 'attention_mask' in inputs and inputs['attention_mask'] is not None and inputs['attention_mask'].ndim == 1:
                    inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0)

            prompt_len = inputs['input_ids'].size(1)

            response_tokens_dict = tokenizer_for_labels(
                response_text + tokenizer_for_labels.eos_token,
                max_length=max_seq_length - prompt_len if max_seq_length > prompt_len else 0,
                truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False
            )
            response_input_ids = response_tokens_dict["input_ids"]

            processed_prompt_ids = inputs['input_ids'].squeeze(0)
            full_input_ids_tensor = torch.cat(
                (processed_prompt_ids, torch.tensor(response_input_ids, dtype=torch.long)), dim=0
            )

            if full_input_ids_tensor.size(0) > max_seq_length:
                full_input_ids_tensor = full_input_ids_tensor[:max_seq_length]

            current_labels_tensor = torch.full_like(full_input_ids_tensor, -100)
            actual_prompt_tokens_in_full = min(processed_prompt_ids.size(0), full_input_ids_tensor.size(0))

            if actual_prompt_tokens_in_full < full_input_ids_tensor.size(0):
                current_labels_tensor[actual_prompt_tokens_in_full:] = full_input_ids_tensor[actual_prompt_tokens_in_full:]

            results["input_ids"][i] = full_input_ids_tensor.tolist()
            results["attention_mask"][i] = torch.ones_like(full_input_ids_tensor).tolist()
            results["labels"][i] = current_labels_tensor.tolist()
            # --- 文本部分处理结束 ---

            # --- 视觉部分处理 (修改) ---
            if inputs.get('pixel_values') is not None:
                pv = inputs['pixel_values']
                if pv.ndim == 2:
                    # 这是预期的 [TotalPatches, FlattenedPatchDim] 格式
                    results["pixel_values"][i] = pv
                    # logger.debug(f"样本 {sample_id_for_log}: Processor 返回了预期的 2D pixel_values, 形状: {pv.shape}")

                    # 同时获取 image_grid_thw (如果存在且与 pixel_values 一起返回)
                    if inputs.get('image_grid_thw') is not None:
                        results["image_grid_thw"][i] = inputs['image_grid_thw']
                        # logger.debug(f"样本 {sample_id_for_log}: 获取了 image_grid_thw, 形状: {inputs['image_grid_thw'].shape}")
                    else:
                         logger.warning(f"样本 {sample_id_for_log}: Processor 返回了 pixel_values 但没有返回 image_grid_thw。模型可能无法正确解析视觉信息。")
                         # 虽然有 pixel_values，但没有 grid 信息可能导致模型出错，可以选择也设为 None
                         # results["pixel_values"][i] = None
                else:
                    # 处理非预期的维度
                    logger.warning(f"样本 {sample_id_for_log}: Processor 返回了非预期的 {pv.ndim}D pixel_values (形状: {pv.shape})。预期为 2D。此样本的 pixel_values 将为 None。")
                    # results["pixel_values"][i] 保持 None
                    # results["image_grid_thw"][i] 保持 None
            else:
                # pixel_values 为 None 的情况 (例如没有输入图像或 processor 处理失败)
                logger.info(f"样本 {sample_id_for_log}: Processor 没有返回 pixel_values。")
                # results["pixel_values"][i] 保持 None
                # results["image_grid_thw"][i] 保持 None
            # --- 视觉部分处理结束 ---

            # 移除 is_valid_sample 的设置
            # results["is_valid_sample"][i] = True

        # 移除 IndexError 和其他 Exception 的捕获后设置 is_valid_sample=False 的逻辑
        # 因为如果 processor 调用失败，已经 continue 了
        # 如果只是图像加载失败，pixel_values 会是 None，样本本身仍然有效

        except Exception as e:
             logger.error(f"样本 {sample_id_for_log} 在预处理过程中发生未预料的严重错误: {e}。")
             # 保持此样本所有字段为空或 None

    # 过滤掉 input_ids 为空的样本 (可能由 processor 调用失败或严重错误导致)
    final_results = {}
    valid_indices = [idx for idx, ids in enumerate(results["input_ids"]) if ids]
    if not valid_indices:
         logger.warning("一个批次处理后没有有效的样本。")
         # 需要返回一个空的字典或符合 map 期望的结构，这里我们返回空字典，map 会处理
         # 注意：如果所有批次都返回空，最终数据集会是空的
         return {}

    for key in results:
         final_results[key] = [results[key][idx] for idx in valid_indices]

    return final_results

# MultimodalDataCollatorForSeq2Seq 保持不变，它应该能处理 pixel_values 为 None 的情况
class MultimodalDataCollatorForSeq2Seq:
    def __init__(self, tokenizer, model=None):
        self.tokenizer = tokenizer
        self.model = model # model 参数可能不再需要，除非用于特定目的

    def __call__(self, features):
        # features 是一个列表，每个元素是 preprocess_function 返回的字典的一个样本

        # --- 文本部分处理 (基本不变) ---
        input_ids_list = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels_list = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # 如果所有样本都处理失败（不太可能，因为 preprocess 已过滤），返回空
        if not input_ids_list:
            return {}

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (padded_input_ids.ne(self.tokenizer.pad_token_id)).long()

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }
        # --- 文本部分处理结束 ---

        # --- 视觉部分处理 (修改) ---
        pixel_values_list = [f.get("pixel_values") for f in features]
        image_grid_thw_list = [f.get("image_grid_thw") for f in features]

        # 检查批次中是否有任何有效的 pixel_values
        if any(pv is not None for pv in pixel_values_list):
            valid_pixel_values = []
            valid_image_grid_thw = []

            for pv, grid in zip(pixel_values_list, image_grid_thw_list):
                if pv is not None and grid is not None:
                    # 确保是 Tensor
                    current_pv_tensor = pv if isinstance(pv, torch.Tensor) else torch.tensor(pv)
                    current_grid_tensor = grid if isinstance(grid, torch.Tensor) else torch.tensor(grid)

                    # 验证维度是否符合预期
                    if current_pv_tensor.ndim == 2 and current_grid_tensor.ndim == 2 and current_grid_tensor.shape[1] == 3:
                         # 验证 patch 数量是否匹配 grid 描述
                         # num_patches_expected = (current_grid_tensor[:, 0] * (current_grid_tensor[:, 1] // self.model.config.vision_config.spatial_merge_size) * (current_grid_tensor[:, 2] // self.model.config.vision_config.spatial_merge_size)).sum().item()
                         # if current_pv_tensor.shape[0] == num_patches_expected: # 这个检查比较复杂，暂时省略
                        valid_pixel_values.append(current_pv_tensor)
                        valid_image_grid_thw.append(current_grid_tensor)
                         # else:
                         #     logger.warning(f"Collator: pixel_values patch count {current_pv_tensor.shape[0]} does not match image_grid_thw description {num_patches_expected}. Skipping visual data for this sample.")
                    else:
                         logger.warning(f"Collator: Skipping visual data due to unexpected shape. pv_ndim={current_pv_tensor.ndim}, grid_ndim={current_grid_tensor.ndim}, grid_shape={current_grid_tensor.shape}")

            # 如果过滤后仍有有效的视觉数据
            if valid_pixel_values:
                # 沿第一个维度（Patch 维度）拼接
                batch_pixel_values = torch.cat(valid_pixel_values, dim=0)
                # 沿第一个维度（Image 维度）拼接
                batch_image_grid_thw = torch.cat(valid_image_grid_thw, dim=0)

                batch["pixel_values"] = batch_pixel_values
                batch["image_grid_thw"] = batch_image_grid_thw
                # logger.debug(f"Collator: Batched pixel_values shape: {batch_pixel_values.shape}")
                # logger.debug(f"Collator: Batched image_grid_thw shape: {batch_image_grid_thw.shape}")

        # 如果批次中没有任何有效的 pixel_values，则 batch 字典中不会包含 pixel_values 和 image_grid_thw 键
        return batch

def main():
    args = parse_args()
    os.makedirs(args.actual_output_dir, exist_ok=True)
    logger.add(args.actual_output_dir / "train.log")
    logger.info(f"解析的参数: {args}")
    logger.info(f"输出目录: {args.actual_output_dir}")
    logger.info(f"图像基本路径: '{args.image_base_path}'")
    if args.skip_sample_if_all_images_missing:
        logger.info("将跳过那些声明有图像但所有图像都无法加载的样本。")


    logger.info(f"从 {args.model_path} 加载处理器和分词器...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.chat_template = USER_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"分词器的 pad_token 未设置。使用 eos_token ({tokenizer.eos_token}) 作为 pad_token。")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            default_pad_token = "<|endoftext|>"
            tokenizer.add_special_tokens({'pad_token': default_pad_token})
            logger.warning(f"未找到 eos_token 或 pad_token。已添加 '{default_pad_token}' 作为 pad_token。")

    if tokenizer.eos_token is None:
        default_eos_token = "<|im_end|>"
        tokenizer.add_special_tokens({'eos_token': default_eos_token})
        logger.warning(f"分词器的 eos_token 未设置。已添加 '{default_eos_token}' 作为 eos_token。")

    # load_and_prepare_dataset 现在只返回初步有效的 Dataset
    initial_dataset = load_and_prepare_dataset(args.dataset_file, args)
    logger.info(f"初步加载的数据集大小（load_and_prepare_dataset后）: {len(initial_dataset)}")


    logger.info("开始预处理数据集 (map)...")
    # tokenized_datasets = DatasetDict() # 不再需要，直接处理 initial_dataset

    # preprocess_function 现在需要 args 来访问 skip_sample_if_all_images_missing
    # functools.partial 可以用来传递额外参数给 map 的函数
    from functools import partial
    preprocess_with_args = partial(preprocess_function, processor=processor, 
                                   max_seq_length=args.max_seq_length, 
                                   tokenizer_for_labels=tokenizer, args=args)

    processed_dataset_with_flags = initial_dataset.map(
        preprocess_with_args,
        batched=True,
        batch_size=10, # 调整批处理大小以平衡内存和速度
        num_proc=min(1, os.cpu_count() or 1), # 图像处理时，num_proc=1 可能更稳定
        # remove_columns 将在 filter 之后进行，或者在定义 trainer 时处理
        # 保留 is_valid_sample 用于过滤
        load_from_cache_file=False # 调试时建议关闭缓存
    )
    logger.info(f"Map 操作后数据集大小（包含无效标记的样本）: {len(processed_dataset_with_flags)}")

    # 过滤掉在 preprocess_function 中标记为无效的样本
    # 确保 is_valid_sample 列存在
    if 'is_valid_sample' not in processed_dataset_with_flags.column_names:
        logger.error("错误： 'is_valid_sample' 列未在 mapped dataset 中找到。无法过滤。")
        final_train_dataset = processed_dataset_with_flags # 或者直接报错退出
    else:
        final_train_dataset = processed_dataset_with_flags.filter(
            lambda example: example['is_valid_sample']
        )
    
    num_actually_processed = len(processed_dataset_with_flags)
    num_skipped_in_map_total = num_actually_processed - len(final_train_dataset)

    logger.info(f"在 preprocess (map) 阶段，共有 {num_skipped_in_map_total} 条数据因处理错误 (IndexError, 所有图像丢失等) 被标记为无效并被过滤。")
    logger.info(f"最终用于训练的数据条数: {len(final_train_dataset)}")

    if len(final_train_dataset) == 0:
        logger.error("没有可用的有效数据进行训练。正在退出。")
        sys.exit(1)
        
    # 从最终数据集中移除 is_valid_sample 和 image_files 列，因为 Trainer 不需要它们
    columns_to_remove_for_trainer = ['is_valid_sample', 'image_files', 'messages', 'response', 
                                     'original_comment_len', 'original_type_len', 'id_in_json']
    # 确保只移除数据集中实际存在的列
    actual_columns_to_remove = [col for col in columns_to_remove_for_trainer if col in final_train_dataset.column_names]
    
    final_train_dataset_for_trainer = final_train_dataset.remove_columns(actual_columns_to_remove)
    logger.info(f"送入Trainer的最终数据集特征: {final_train_dataset_for_trainer.features}")


    # --- 调试代码部分 ---
    logger.info("--- 开始调试过滤后的预处理样本 (最终训练集中的第一个样本) ---")
    if len(final_train_dataset_for_trainer) > 0 :
        # 注意：这里的 full_dataset 变量名可能需要调整，因为它现在代表过滤后的数据集
        # 我们从 final_train_dataset (包含原始信息的版本) 中取一个样本来展示
        
        debug_sample_original_info_idx = 113 # 假设我们想看过滤后数据集的第一个有效样本
                                        # 它在 initial_dataset 中的原始信息可以通过 id_in_json 追溯 (如果需要)
        
        # 从处理后的数据集中获取 input_ids 和 labels
        # processed_sample_for_debug = final_train_dataset_for_trainer[debug_sample_original_info_idx] # 这是给Trainer的
        # 为了调试，我们可能想看过滤前、包含is_valid_sample的那个数据集里的内容
        # 但为了简单，我们直接用 final_train_dataset_for_trainer 的样本
        if len(final_train_dataset_for_trainer) > debug_sample_original_info_idx:
            processed_sample_for_debug = final_train_dataset_for_trainer[debug_sample_original_info_idx]
            input_ids_debug = processed_sample_for_debug['input_ids']
            labels_debug = processed_sample_for_debug['labels']

            if isinstance(input_ids_debug, list): input_ids_debug_tensor = torch.tensor(input_ids_debug)
            else: input_ids_debug_tensor = input_ids_debug
            if isinstance(labels_debug, list): labels_debug_tensor = torch.tensor(labels_debug)
            else: labels_debug_tensor = labels_debug

            logger.info(f"调试样本索引 {debug_sample_original_info_idx} - Input IDs 长度: {len(input_ids_debug_tensor)}")
            decoded_input_text = tokenizer.decode(input_ids_debug_tensor, skip_special_tokens=False)
            logger.info(f"调试样本索引 {debug_sample_original_info_idx} - 解码后的 Input IDs (可能包含特殊标记): '{decoded_input_text[:500]}...'")
            
            valid_label_ids_for_decode = [lbl for lbl in labels_debug_tensor.tolist() if lbl != -100]
            logger.info(f"调试样本索引 {debug_sample_original_info_idx} - 有效目标标记数量: {len(valid_label_ids_for_decode)}")
            if valid_label_ids_for_decode:
                decoded_valid_labels_text = tokenizer.decode(valid_label_ids_for_decode)
                logger.info(f"调试样本索引 {debug_sample_original_info_idx} - 解码后的有效 Labels: '{decoded_valid_labels_text}'")

            if 'pixel_values' in processed_sample_for_debug and processed_sample_for_debug['pixel_values'] is not None:
                pv_debug = processed_sample_for_debug['pixel_values']
                # pv_debug 此时应该是 collator 处理前的单个样本的 tensor [NumImg, C, H, W] 或 None
                if isinstance(pv_debug, torch.Tensor):
                     logger.info(f"调试样本索引 {debug_sample_original_info_idx} - Pixel Values 形状: {pv_debug.shape}, 类型: {pv_debug.dtype}")
                elif isinstance(pv_debug, list) and pv_debug : # 如果还是列表（不太可能在这个阶段）
                     logger.info(f"调试样本索引 {debug_sample_original_info_idx} - Pixel Values 是列表，首元素形状: {torch.tensor(pv_debug[0]).shape if pv_debug else 'N/A'}")
                else:
                     logger.info(f"调试样本索引 {debug_sample_original_info_idx} - Pixel Values 是 None 或空列表。")

            else:
                logger.info(f"调试样本索引 {debug_sample_original_info_idx} - 此样本中没有 pixel_values。")
        else:
            logger.warning("最终训练数据集为空或数量不足以进行样本调试。")

    else:
        logger.warning("无法执行样本调试，因为最终训练数据集为空。")
    logger.info("--- 预处理样本调试结束 ---")
    # --- 调试代码结束 ---


    logger.info(f"从 {args.model_path} 加载基础模型...")
    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA enabled, using 4-bit quantization.")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not args.use_qlora and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", trust_remote_code=True
    )
    
    # if model.config.vocab_size != len(tokenizer):
    #     logger.warning(f"模型词汇表大小 ({model.config.vocab_size}) != 分词器词汇表大小 ({len(tokenizer)})。正在调整模型嵌入的大小。")
    #     model.resize_token_embeddings(len(tokenizer))

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info("模型已准备好进行 k-bit 训练 (QLoRA)。")

    target_modules = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i in range(len(model.model.layers)):
            target_modules.extend([
                f"model.layers.{i}.self_attn.q_proj", f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj", f"model.layers.{i}.self_attn.o_proj",
                f"model.layers.{i}.mlp.gate_proj", f"model.layers.{i}.mlp.up_proj",
                f"model.layers.{i}.mlp.down_proj",
            ])
    if not target_modules: 
        logger.warning("无法根据 'model.layers' 自动检测 LoRA 目标模块。使用通用的 Qwen 列表。请验证！")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    logger.info(f"尝试使用 LoRA 目标模块: {target_modules[:3]}... 等。")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )

    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"应用 PEFT LoRA 配置失败: {e}")
        logger.error("请仔细检查您的模型架构 (Qwen2.5-VL) 的 `lora_target_modules`。")
        sys.exit(1)

    training_args = TrainingArguments(
        output_dir=str(args.actual_output_dir), run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=not (args.use_qlora or torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_bf16_supported() and not args.use_qlora,
        eval_strategy="no", save_total_limit=3,
        load_best_model_at_end=False, report_to="tensorboard",
        remove_unused_columns=False, # Trainer会处理remove_columns，这里设为False，因为我们已手动处理
        label_names=["labels"],
        gradient_checkpointing=True, # Enable if using QLoRA for memory saving
        gradient_checkpointing_kwargs={'use_reentrant':False},
        # gradient_checkpointing=args.use_qlora,
        # gradient_checkpointing_kwargs={'use_reentrant':False} if args.use_qlora else None, # 仅当GC启用时传递
    )

    data_collator = MultimodalDataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=final_train_dataset_for_trainer, # 使用过滤并移除了多余列的数据集
        eval_dataset=None,
        data_collator=data_collator, 
        tokenizer=processor 
    )
    
    logger.info("开始训练...")
    try:
        logger.info(f"将使用 {len(final_train_dataset_for_trainer)} 条有效数据进行训练。")
        train_result = trainer.train()

        logger.info("保存最终模型 (LoRA 适配器权重)...")
        final_model_dir_str = str(args.actual_output_dir / "final_checkpoint")
        trainer.save_model(final_model_dir_str)
        processor.save_pretrained(final_model_dir_str)
        logger.info(f"处理器和最终 LoRA 适配器权重已保存到 {final_model_dir_str}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.exception("训练期间发生错误。")
        if isinstance(e, torch.cuda.OutOfMemoryError): 
            logger.error(
                "CUDA 内存不足。"
                "尝试减少 `per_device_train_batch_size`、`max_seq_length`，"
                "或启用 QLoRA (`--use_qlora`) 或梯度累积。"
            )
        raise
    logger.info("训练完成。")


if __name__ == "__main__":
    main()