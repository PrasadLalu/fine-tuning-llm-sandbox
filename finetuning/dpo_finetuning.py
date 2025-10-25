"""
DPO (Direct Preference Optimization) Fine-tuning Script for Finance Dataset
Uses preference pairs to align model with human preferences
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer
import os

def load_finance_dpo_dataset(file_path):
    """Load and prepare Finance DPO dataset with preference pairs"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    chosen = []
    rejected = []
    
    for item in data:
        prompt = f"### Question: {item['prompt']}\n### Answer:"
        prompts.append(prompt)
        chosen.append(item['chosen'])
        rejected.append(item['rejected'])
    
    return Dataset.from_dict({
        'prompt': prompts,
        'chosen': chosen,
        'rejected': rejected
    })

def train_dpo_finetuning(
    dataset_path='datasets/finance_dpo_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/finance_dpo_finetuned',
    epochs=3,
    batch_size=2,
    learning_rate=5e-6,
    beta=0.1
):
    """
    DPO fine-tuning function for Finance dataset
    
    Args:
        dataset_path: Path to Finance DPO dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        beta: DPO beta parameter (KL divergence weight)
    """
    
    print("=" * 50)
    print("DPO FINE-TUNING - FINANCE DATASET")
    print("=" * 50)
    
    # Load tokenizer and model
    print(f"\n1. Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto' if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Reference model (frozen copy)
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto' if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    if torch.cuda.is_available():
        model = model.half()
        model_ref = model_ref.half()
    
    print(f"   Model loaded with {model.num_parameters():,} parameters")
    
    # Load dataset
    print(f"\n2. Loading DPO dataset from: {dataset_path}")
    dataset = load_finance_dpo_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} preference pairs")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        save_strategy='epoch',
        logging_steps=10,
        warmup_steps=50,
        optim='adamw_torch',
        save_total_limit=2,
        report_to='none',
        remove_unused_columns=False
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=beta,
        max_prompt_length=256,
        max_length=512
    )
    
    # Train
    print("\n3. Starting DPO fine-tuning...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Beta (KL weight): {beta}")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("\n" + "-" * 50)
    
    dpo_trainer.train()
    
    # Save model
    print(f"\n4. Saving model to: {output_dir}")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 50)
    print("DPO FINE-TUNING COMPLETED!")
    print("=" * 50)
    
    return output_dir

if __name__ == "__main__":
    output_path = train_dpo_finetuning()
    print(f"\nModel saved at: {output_path}")

