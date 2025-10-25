"""
Full Fine-tuning Script for HR Dataset
Uses TinyLlama-1.1B model with complete parameter training
"""

import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

def load_hr_dataset(file_path):
    """Load and prepare HR dataset"""
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

def train_full_finetuning(
    dataset_path='datasets/hr_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/hr_full_finetuned',
    epochs=3,
    batch_size=4,
    learning_rate=2e-5
):
    """
    Full fine-tuning function for HR dataset
    
    Args:
        dataset_path: Path to HR dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    
    print("=" * 50)
    print("FULL FINE-TUNING - HR DATASET")
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
    
    print(f"   Model loaded with {model.num_parameters():,} parameters")
    
    # Load and prepare dataset
    print(f"\n2. Loading dataset from: {dataset_path}")
    dataset = load_hr_dataset(dataset_path)
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
        warmup_steps=100,
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
    print("\n4. Starting full fine-tuning...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("\n" + "-" * 50)
    
    trainer.train()
    
    # Save model
    print(f"\n5. Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 50)
    print("FULL FINE-TUNING COMPLETED!")
    print("=" * 50)
    
    return output_dir

if __name__ == "__main__":
    output_path = train_full_finetuning()
    print(f"\nModel saved at: {output_path}")

