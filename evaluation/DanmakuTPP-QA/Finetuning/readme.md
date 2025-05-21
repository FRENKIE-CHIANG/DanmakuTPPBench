# Qwen2.5-VL-Finetuning

Finetuning Qwen2.5-VL based on DanmakuTPP-QA.

## Overview

This project utilizes the Qwen2.5-VL model with LoRA fine-tuning to work on 10 tasks.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- Other dependencies listed in `requirements.txt`

## Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train/train_task1.py \
    --base_model_path "path to your base model" \
    --train_dataset_file "path to your training dataset" \
    --output_dir "path to save outputs"
```

### Inference

```bash
python test/test_task1.py \
    --base_model_path "path to your base model" \
    --lora_adapter_path "path to your LoRA adapter" \
    --test_dataset_file "path to your test dataset" \
    --output_file "path to save test results"
```