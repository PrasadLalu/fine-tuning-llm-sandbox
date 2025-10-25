#!/bin/bash
# H100 Training Commands for All Model Types
# Run these on your H100 GPU server

echo "🚀 Starting H100 Training Pipeline"

# Set environment variables for H100 optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Create models directory
mkdir -p models

echo "📊 Training HR Model (Full Fine-tuning)..."
python finetuning/full_finetuning.py \
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dataset_path "datasets/hr_dataset.json" \
    --output_dir "models/hr_full_finetuned" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 100

echo "💰 Training Finance Model (DPO)..."
python finetuning/dpo_finetuning.py \
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dataset_path "datasets/finance_dpo_dataset.json" \
    --output_dir "models/finance_dpo_finetuned" \
    --num_epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --beta 0.1

echo "🛒 Training Sales Model (PEFT)..."
python finetuning/peft_finetuning.py \
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dataset_path "datasets/sales_dataset.json" \
    --output_dir "models/sales_peft_finetuned" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --target_modules "q_proj,v_proj"

echo "🏥 Training Healthcare Model (LoRA)..."
python finetuning/lora_finetuning.py \
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dataset_path "datasets/healthcare_dataset.json" \
    --output_dir "models/healthcare_lora_finetuned" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32

echo "📢 Training Marketing Model (QLoRA)..."
python finetuning/qlora_finetuning.py \
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dataset_path "datasets/marketing_dataset.json" \
    --output_dir "models/marketing_qlora_finetuned" \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --quantization_config "4bit"

echo "✅ All models trained successfully!"
echo "📁 Models saved in: models/"
