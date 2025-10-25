#!/usr/bin/env python3
"""
Master Training Script - Train All Models
Run this script to train all 5 models sequentially
"""

import subprocess
import sys
import os
import time

def run_training_script(script_name, description):
    """Run a training script and handle errors"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, 
            f"finetuning/{script_name}"
        ], check=True, capture_output=True, text=True)
        
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🎯 Master Training Script - All Models")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("finetuning"):
        print("❌ Error: finetuning directory not found!")
        print("Make sure you're in the project root directory.")
        return
    
    # Training sequence
    training_scripts = [
        ("full_finetuning.py", "HR Model (Full Fine-tuning)"),
        ("lora_finetuning.py", "Healthcare Model (LoRA)"),
        ("peft_finetuning.py", "Sales Model (PEFT)"),
        ("qlora_finetuning.py", "Marketing Model (QLoRA)"),
        ("dpo_finetuning.py", "Finance Model (DPO)")
    ]
    
    successful = 0
    failed = 0
    
    for script, description in training_scripts:
        success = run_training_script(script, description)
        if success:
            successful += 1
        else:
            failed += 1
        
        # Wait a bit between trainings
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("📊 Training Summary")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    if successful == len(training_scripts):
        print("\n🎉 All models trained successfully!")
        print("📁 Models saved in: models/")
    else:
        print(f"\n⚠️  {failed} model(s) failed. Check the errors above.")

if __name__ == "__main__":
    main()
