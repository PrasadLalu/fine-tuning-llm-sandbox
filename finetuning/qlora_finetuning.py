"""
QLoRA (Quantized LoRA) Fine-tuning Script for Marketing Dataset
Memory-efficient fine-tuning using 4-bit quantization + LoRA
"""

import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os

def load_marketing_dataset(file_path):
    """Load and prepare Marketing dataset"""
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

def train_qlora_finetuning(
    dataset_path='datasets/marketing_dataset.json',
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir='models/marketing_qlora_finetuned',
    epochs=5,
    batch_size=4,
    learning_rate=2e-4,
    lora_r=16,
    lora_alpha=32
):
    """
    QLoRA fine-tuning function for Marketing dataset
    
    Args:
        dataset_path: Path to Marketing dataset JSON
        model_name: Base model name
        output_dir: Output directory for model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
    """
    
    print("=" * 50)
    print("QLoRA FINE-TUNING - MARKETING DATASET")
    print("=" * 50)
    
    # Load tokenizer
    print(f"\n1. Loading model with 4-bit quantization: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization configuration (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map='auto' if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    if torch.cuda.is_available() and bnb_config is None:
        model = model.half()
    
    # Prepare model for k-bit training
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
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
    
    print(f"   Model loaded with 4-bit quantization")
    print(f"   Base model parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   LoRA rank: {lora_r}, LoRA alpha: {lora_alpha}")
    
    # Load and prepare dataset
    print(f"\n2. Loading dataset from: {dataset_path}")
    dataset = load_marketing_dataset(dataset_path)
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
        optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
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
    print("\n4. Starting QLoRA fine-tuning...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Quantization: 4-bit")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("\n" + "-" * 50)
    
    trainer.train()
    
    # Save model
    print(f"\n5. Saving QLoRA model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 50)
    print("QLoRA FINE-TUNING COMPLETED!")
    print("=" * 50)
    
    return output_dir

if __name__ == "__main__":
    output_path = train_qlora_finetuning()
    print(f"\nModel saved at: {output_path}")

