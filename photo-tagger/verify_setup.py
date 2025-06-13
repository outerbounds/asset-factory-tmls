#!/usr/bin/env python3

import os
import sys

def check_files():
    """Verify core files exist"""
    required_files = [
        'finetune_vlm.py',
        'prepare_vlm_data.py', 
        'prompter.py',
        'promptmodels.py',
        'test_pipeline.py',
        'check_data.py',
        'run_tests.sh'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All core files present")
    return True

def check_dependencies():
    """Check if key dependencies can be imported"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed") 
        return False
    
    try:
        import peft
        print(f"✅ PEFT: {peft.__version__}")
    except ImportError:
        print("❌ PEFT not installed")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")
        return False
    
    return True

def check_config_files():
    """Check configuration files"""
    if os.path.exists('prompt.txt'):
        with open('prompt.txt', 'r') as f:
            prompt = f.read().strip()
        print(f"✅ Prompt file: {len(prompt)} chars")
    else:
        print("⚠️  prompt.txt missing")
    
    if os.path.exists('obhost.txt'):
        with open('obhost.txt', 'r') as f:
            obhost = f.read().strip()
        print(f"✅ Outerbounds host: {obhost[:20]}...")
    else:
        print("⚠️  obhost.txt missing")

def main():
    print("🔧 VLM Factory Setup Verification")
    print("=" * 40)
    
    files_ok = check_files()
    deps_ok = check_dependencies()
    check_config_files()
    
    print("\n" + "=" * 40)
    if files_ok and deps_ok:
        print("✅ Setup verified - ready for VLM Factory pipeline")
        print("\nNext steps:")
        print("1. Configure Outerbounds: outerbounds configure")
        print("2. Run data preparation: python prepare_vlm_data.py run")
        print("3. Run fine-tuning: python finetune_vlm.py run --epochs 1")
        print("4. Test pipeline: ./run_tests.sh")
    else:
        print("❌ Setup incomplete - install missing dependencies")
        print("\nInstall command:")
        print("pip install transformers torch peft datasets accelerate")

if __name__ == "__main__":
    main() 