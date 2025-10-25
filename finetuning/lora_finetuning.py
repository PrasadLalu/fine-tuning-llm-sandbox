"""
LoRA (Low-Rank Adaptation) Fine-tuning Script for Healthcare Dataset
Efficient fine-tuning using low-rank matrices
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os

def load_healthcare_dataset(file_path):
    """Load and prepare Healthcare dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_dict({
        'text': [f"### Instruction: {item['instruction']}\n### Response: {item['output']}" 
                 for item in data]
    })

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize dataset"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )

def train_lora_finetuning(
    dataset_path='datasets/healthcare_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/healthcare_lora_finetuned',
    epochs=5,
    batch_size=4,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32
):
    """
    LoRA fine-tuning function for Healthcare dataset
    
    Args:
        dataset_path: Path to Healthcare dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
    """
    
    print("=" * 50)
    print("LoRA FINE-TUNING - HEALTHCARE DATASET")
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
    
    if torch.cuda.is_available():
        model = model.half()
    
    # Apply LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"   Base model parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   LoRA rank: {lora_r}, LoRA alpha: {lora_alpha}")
    
    # Load and prepare dataset
    print(f"\n2. Loading dataset from: {dataset_path}")
    dataset = load_healthcare_dataset(dataset_path)
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Tokenize dataset
    print("\n3. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        save_strategy='epoch',
        logging_steps=10,
        warmup_steps=50,
        optim='adamw_torch',
        save_total_limit=2,
        report_to='none'
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train
    print("\n4. Starting LoRA fine-tuning...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("\n" + "-" * 50)
    
    trainer.train()
    
    # Save model
    print(f"\n5. Saving LoRA model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 50)
    print("LoRA FINE-TUNING COMPLETED!")
    print("=" * 50)
    
    return output_dir

if __name__ == "__main__":
    output_path = train_lora_finetuning()
    print(f"\nModel saved at: {output_path}")

