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
    Qwen2_5_VLForConditionalGeneration, 
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from tqdm.auto import tqdm


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
    parser = argparse.ArgumentParser(description="微调一个大型语言模型（基于Qwen VL）用于预测弹幕波峰任务。")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Hugging Face Hub上的基础模型路径或本地路径。")
    parser.add_argument("--dataset_file", type=str, default=DEFAULT_DATA_FILE, help="JSON数据文件路径 (例如, test.json)。")
    parser.add_argument("--output_dir", type=str, default="./results_danmaku_sft_vl", help="训练好的模型和日志的输出目录。")
    parser.add_argument("--run_name_suffix", type=str, default="_danmaku_sft_vl", help="输出目录中运行名称的后缀。")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每GPU的训练批次大小。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率。")
    parser.add_argument("--max_seq_length", type=int, default=8182, help="分词后的最大序列长度。")
    parser.add_argument("--logging_steps", type=int, default=10, help="每X步记录一次日志。")
    parser.add_argument("--save_steps", type=int, default=100, help="每X步保存一次检查点。")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA的r参数。")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA的alpha参数。")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA的dropout率。")
    parser.add_argument("--use_qlora", action='store_true', help="启用QLoRA (4位量化)。")
    # parser.add_argument("--train_on_dev", action='store_true', help="使用开发集进行训练 (用于调试或数据量较少时)。") # 此参数不再需要
    parser.add_argument("--max_raw_comment_seq_len", type=int, default=200, # 您脚本中设置的默认值
                        help="输入到prompt中的原始弹幕时间序列的最大长度（条目数）。-1表示不截断。")
    args = parser.parse_args()
    args.run_name = Path(args.model_path).name + args.run_name_suffix
    args.actual_output_dir = Path(args.output_dir) / args.run_name
    return args

def load_and_prepare_dataset(dataset_file_path, tokenizer, args):
    logger.info(f"从 {dataset_file_path} 加载数据集...")
    try:
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"加载或解析JSON文件失败: {e}")
        sys.exit(1)

    if not isinstance(raw_data, list):
        logger.error("数据集应为一个字典列表。")
        sys.exit(1)

    processed_data = {'prompt': [], 'response': [], 'original_comment_len': []}
    MAX_COMMENT_SEQUENCE_LENGTH = args.max_raw_comment_seq_len

    for item in tqdm(raw_data, desc="构建prompt和response"):
        original_comment_sequence = item.get("comment_time_sequence", [])
        processed_data['original_comment_len'].append(len(original_comment_sequence))

        if MAX_COMMENT_SEQUENCE_LENGTH != -1 and len(original_comment_sequence) > MAX_COMMENT_SEQUENCE_LENGTH:
            truncated_comment_sequence = original_comment_sequence[-MAX_COMMENT_SEQUENCE_LENGTH:]
        else:
            truncated_comment_sequence = original_comment_sequence

        comment_seq_str = ", ".join(map(str, truncated_comment_sequence))
        question_str = item.get("question", "")
        num_peaks_str = str(item.get("num_peaks", ""))
        peak_ts_str = ", ".join(map(str, item.get("peak_timestamps", [])))

        user_content = f"弹幕时间序列: [{comment_seq_str}]\n\n问题: {question_str}"
        assistant_content = f"波峰数量为 {num_peaks_str}。波峰时间戳为: [{peak_ts_str}]。"

        messages_for_prompt = [
            {"role": "system", "content": "你是一个有用的助手，我需要你进行时间点序列的分析"},
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
        logger.error("未能从数据文件中提取任何样本，请检查数据格式。")
        sys.exit(1)
    
    dataset = Dataset.from_dict(processed_data)
    
    # --- 修改：不再划分验证集，全部用于训练 ---
    all_train_dataset = DatasetDict({'train': dataset})
    logger.info(f"所有数据 ({len(dataset)}) 将用于训练。")
    # --- 结束修改 ---
    
    logger.info(f"训练集大小: {len(all_train_dataset['train'])}")
    # logger.info(f"验证集大小: 0") # 不再有验证集
    
    return all_train_dataset # 返回修改后的DatasetDict

def preprocess_function(examples, tokenizer, max_seq_length):
    inputs_ids_list = []
    labels_list = []

    for prompt, response in zip(examples['prompt'], examples['response']):
        text_all = prompt + response + tokenizer.eos_token

        tokenized_all = tokenizer(
            text_all,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        tokenized_prompt = tokenizer(
            prompt,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        prompt_len = len(tokenized_prompt["input_ids"])

        current_labels = tokenized_all["input_ids"].copy()
        current_labels[:prompt_len] = [-100] * prompt_len

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
    logger.info(f"解析得到的参数: {args}")
    logger.info(f"输出目录: {args.actual_output_dir}")

    logger.info(f"从 {args.model_path} 加载处理器和分词器...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.chat_template = USER_CHAT_TEMPLATE

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"分词器的 pad_token 未设置，将使用 eos_token ({tokenizer.eos_token}) 作为 pad_token。")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            default_pad_token = "<|endoftext|>"
            tokenizer.add_special_tokens({'pad_token': default_pad_token})
            logger.warning(f"eos_token 和 pad_token 均未找到，已添加 '{default_pad_token}' 作为 pad_token。请检查模型兼容性。")

    if tokenizer.eos_token is None:
        default_eos_token = "<|im_end|>"
        tokenizer.add_special_tokens({'eos_token': default_eos_token})
        logger.warning(f"分词器的 eos_token 未设置，已添加 '{default_eos_token}' 作为 eos_token。")

    full_dataset = load_and_prepare_dataset(args.dataset_file, tokenizer, args) # full_dataset 现在是 DatasetDict({'train': ...})

    logger.info("预处理数据集...")
    tokenized_datasets = DatasetDict()

    # --- 修改：只处理训练集 ---
    # for split, dataset_split in full_dataset.items(): # 原来的循环
    # 只处理 'train' split
    if 'train' in full_dataset:
        dataset_split = full_dataset['train']
        tokenized_datasets['train'] = dataset_split.map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_seq_length),
            batched=True,
            batch_size=1000,
            num_proc=min(4, os.cpu_count() or 1),
            remove_columns=[col for col in dataset_split.column_names if col not in ['input_ids', 'labels']]
        )
    else:
        logger.error("加载的数据集中没有 'train' 部分，无法继续。")
        sys.exit(1)
    # --- 结束修改 ---

    logger.info(f"分词后的训练集特征: {tokenized_datasets['train'].features}")
    # if 'dev' in tokenized_datasets and len(tokenized_datasets['dev']) > 0: # 不再有dev集
    #      logger.info(f"分词后的验证集特征: {tokenized_datasets['dev'].features}")

    # --- DEBUGGING SECTION ---
    logger.info("--- 开始调试预处理样本 (取训练集第一个样本) ---")
    # full_dataset['train'] 包含原始的 prompt, response, original_comment_len
    # tokenized_datasets['train'] 包含处理后的 input_ids, labels
    if len(full_dataset['train']) > 0 and len(tokenized_datasets['train']) > 0:
        raw_sample_idx = 0
        
        raw_prompt_text_example = full_dataset['train'][raw_sample_idx]['prompt']
        raw_response_text_example = full_dataset['train'][raw_sample_idx]['response']
        original_comment_len_example = full_dataset['train'][raw_sample_idx].get('original_comment_len', 'N/A')

        logger.info(f"样本 {raw_sample_idx} - 原始弹幕列表长度 (截断前): {original_comment_len_example}")
        logger.info(f"样本 {raw_sample_idx} - 用于生成prompt的文本 (部分): {raw_prompt_text_example[:]}...")
        logger.info(f"样本 {raw_sample_idx} - 用于生成response的文本: {raw_response_text_example}")

        text_all_debug = raw_prompt_text_example + raw_response_text_example + tokenizer.eos_token
        tokenized_all_debug = tokenizer(
            text_all_debug, max_length=args.max_seq_length, truncation=True, padding=False
        )
        input_ids_debug = tokenized_all_debug["input_ids"]
        final_tokenized_length = len(input_ids_debug)
        logger.info(f"样本 {raw_sample_idx} - 'text_all' (prompt+response+eos) 分词后并截断到 {args.max_seq_length} 的总长度: {final_tokenized_length}")

        tokenized_prompt_debug = tokenizer(
            raw_prompt_text_example, max_length=args.max_seq_length, truncation=True, padding=False
        )
        prompt_len_debug = len(tokenized_prompt_debug["input_ids"])
        logger.info(f"样本 {raw_sample_idx} - 'prompt_text' 单独分词后并截断到 {args.max_seq_length} 的长度 (prompt_len): {prompt_len_debug}")
        
        tokenized_response_debug_only = tokenizer(raw_response_text_example, add_special_tokens=False)
        response_tokens_len_approx = len(tokenized_response_debug_only["input_ids"])
        logger.info(f"样本 {raw_sample_idx} - 'response_text' 单独分词后的大致长度 (不含eos): {response_tokens_len_approx}")

        processed_sample_labels = tokenized_datasets['train'][raw_sample_idx]['labels']
        valid_label_count = sum(1 for lbl in processed_sample_labels if lbl != -100)
        logger.info(f"样本 {raw_sample_idx} - 最终 Labels 长度: {len(processed_sample_labels)}")
        logger.info(f"样本 {raw_sample_idx} - 最终 Labels 中有效标签数量 (非 -100): {valid_label_count}")

        if prompt_len_debug >= final_tokenized_length and final_tokenized_length > 0 :
             logger.warning(f"样本 {raw_sample_idx} - 警告: prompt_len ({prompt_len_debug}) >= 最终序列总长 ({final_tokenized_length})!")
             logger.warning(f"样本 {raw_sample_idx} - 这意味着可能没有空间给response，或者response完全被视为prompt的一部分被掩码。")
        elif final_tokenized_length <= prompt_len_debug :
             logger.warning(f"样本 {raw_sample_idx} - 警告: 最终序列总长 ({final_tokenized_length}) <= prompt_len ({prompt_len_debug})!")
             logger.warning(f"样本 {raw_sample_idx} - 这同样意味着可能没有有效response部分用于学习。")

        if valid_label_count > 0:
            logger.info(f"样本 {raw_sample_idx} - Labels (部分，前50个): {processed_sample_labels[:50]}")
            logger.info(f"样本 {raw_sample_idx} - Labels (部分，后50个): {processed_sample_labels[-50:]}")
            valid_label_ids_for_decode = [l for l in processed_sample_labels if l != -100]
            decoded_valid_labels_text = tokenizer.decode(valid_label_ids_for_decode)
            logger.info(f"样本 {raw_sample_idx} - 解码后的有效 Labels 部分应预测文本: '{decoded_valid_labels_text}'")
        else:
            logger.info(f"样本 {raw_sample_idx} - Labels 全是 -100 或为空，没有有效的学习目标。")
            input_ids_from_processed = tokenized_datasets['train'][raw_sample_idx]['input_ids']
            decoded_input_from_processed = tokenizer.decode(input_ids_from_processed)
            logger.info(f"样本 {raw_sample_idx} - (参考)解码后的实际input_ids: '{decoded_input_from_processed}'")
    else:
        logger.warning("无法执行样本调试，因为训练集为空。")
    logger.info("--- 结束调试预处理样本 ---")

    logger.info(f"从 {args.model_path} 加载基础模型...")
    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        logger.info("QLoRA已启用，使用4位量化。")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not args.use_qlora and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", trust_remote_code=True
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        logger.info("模型已为k-bit训练准备就绪 (QLoRA)。")

    example_target_modules = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        for i in range(num_layers):
            example_target_modules.extend([
                f"model.layers.{i}.self_attn.q_proj", f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj", f"model.layers.{i}.self_attn.o_proj",
                f"model.layers.{i}.mlp.gate_proj", f"model.layers.{i}.mlp.up_proj",
                f"model.layers.{i}.mlp.down_proj",
            ])
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
         num_layers = len(model.transformer.h)
         for i in range(num_layers):
            example_target_modules.extend([
                f"transformer.h.{i}.attn.c_attn", f"transformer.h.{i}.attn.c_proj",
                f"transformer.h.{i}.mlp.w1", f"transformer.h.{i}.mlp.w2",
                f"transformer.h.{i}.mlp.c_proj",
            ])
    if not example_target_modules:
        logger.warning("未能自动检测LoRA目标模块，使用通用回退列表。请务必检查并修改 target_modules！")
        example_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=example_target_modules
    )
    logger.info(f"LoRA target modules (示例，请核实): {example_target_modules[:3]}... 等")

    try:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"应用PEFT LoRA配置失败: {e}")
        logger.error("请仔细检查 `lora_target_modules` 是否与您的模型结构匹配。")
        logger.error("您可以通过打印 `model.named_modules()` 查看所有模块名。")
        sys.exit(1)

    # --- 修改 TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=str(args.actual_output_dir), run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=not (args.use_qlora or torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_bf16_supported() and not args.use_qlora,
        eval_strategy="no", # 不进行评估
        # eval_steps=None, # 因为不评估，所以不需要设置
        save_total_limit=3,
        load_best_model_at_end=False, # 不加载最佳模型，因为没有评估
        report_to="tensorboard", remove_unused_columns=False, label_names=["labels"],
        # gradient_checkpointing=args.use_qlora,
        gradient_checkpointing=True, # Enable if using QLoRA for memory saving
        gradient_checkpointing_kwargs={'use_reentrant':False},
    )
    # --- 结束修改 ---

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # --- 修改 Trainer 初始化 ---
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=None, # 不使用验证集
        data_collator=data_collator, tokenizer=tokenizer
    )
    # --- 结束修改 ---

    logger.info("开始训练...")
    train_dataset_obj = tokenized_datasets.get("train")
    if not train_dataset_obj or len(train_dataset_obj) == 0:
         logger.error("没有可用的训练数据，程序退出。"); sys.exit(1)
    try:
        logger.info(f"将在 {len(train_dataset_obj)} 个样本上进行训练。")
        # if training_args.do_eval and tokenized_datasets.get("dev") and len(tokenized_datasets['dev']) > 0: # 不再有dev集相关的日志
        #     logger.info(f"将在 {len(tokenized_datasets['dev'])} 个样本上进行评估。")

        train_result = trainer.train()

        logger.info("保存最终模型 (LoRA适配器权重)...")
        final_model_dir_str = str(args.actual_output_dir / "final_checkpoint")
        trainer.save_model(final_model_dir_str)
        processor.save_pretrained(final_model_dir_str)
        logger.info(f"处理器和最终的LoRA适配器已保存到 {final_model_dir_str}")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics); trainer.save_metrics("train", metrics); trainer.save_state()
    except Exception as e:
        logger.exception("训练过程中发生错误。");
        raise
    logger.info("训练完成。")

if __name__ == "__main__":
    main()