#!/usr/bin/env python3
"""
Setup script for IndicTrans2 environment
This script installs dependencies and downloads models
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("="*60)
    print("Installing Dependencies")
    print("="*60)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("Error: requirements.txt not found!")
        return False
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r {requirements_file}", 
        "Installing required packages"
    )
    
    if success:
        print("\n✓ All dependencies installed successfully!")
    return success

def download_models():
    """Download IndicTrans2 models"""
    print("\n" + "="*60)
    print("Downloading Models")
    print("="*60)
    
    models_to_download = [
        "ai4bharat/indictrans2-en-indic-dist-200M",
        "ai4bharat/indictrans2-indic-en-dist-200M",
    ]
    
    download_script = '''
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download
import sys

model_name = sys.argv[1]
print(f"Downloading {model_name}...")

try:
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Tokenizer downloaded")
    
    # Download model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float32  # Use float32 for CPU compatibility
    )
    print("✓ Model downloaded")
    
    print(f"✓ {model_name} successfully downloaded and cached!")
    
except Exception as e:
    print(f"✗ Error downloading {model_name}: {e}")
    sys.exit(1)
'''
    
    # Create temporary download script
    download_script_path = Path(__file__).parent / "temp_download.py"
    with open(download_script_path, 'w') as f:
        f.write(download_script)
    
    all_success = True
    for model_name in models_to_download:
        print(f"\nDownloading {model_name}...")
        success = run_command(
            f"{sys.executable} {download_script_path} {model_name}", 
            f"Downloading {model_name}"
        )
        if not success:
            all_success = False
    
    # Clean up temporary script
    try:
        download_script_path.unlink()
    except:
        pass
    
    return all_success

def check_installation():
    """Check if installation was successful"""
    print("\n" + "="*60)
    print("Checking Installation")
    print("="*60)
    
    test_script = '''
try:
    import torch
    print("✓ PyTorch imported successfully")
    
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print("✓ Transformers imported successfully")
    
    from IndicTransToolkit.processor import IndicProcessor
    print("✓ IndicTransToolkit imported successfully")
    
    # Test model loading (just check if it can be found)
    model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✓ Model can be loaded from cache")
    
    print("\\n🎉 Installation successful! You can now run test_indictrans.py")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please check your installation")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Models might not be downloaded properly")
'''
    
    test_script_path = Path(__file__).parent / "temp_test.py"
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    
    success = run_command(
        f"{sys.executable} {test_script_path}", 
        "Testing installation"
    )
    
    # Clean up
    try:
        test_script_path.unlink()
    except:
        pass
    
    return success

def main():
    """Main setup function"""
    print("IndicTrans2 Setup Script")
    print("This will install dependencies and download models")
    print("This may take several minutes and requires internet connection")
    
    response = input("\nDo you want to continue? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Setup cancelled.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check the errors above.")
        return
    
    # Download models
    print("\nNote: Model download may take 5-10 minutes depending on your internet speed.")
    download_response = input("Download models now? (Y/n): ").lower().strip()
    
    if download_response not in ['n', 'no']:
        if not download_models():
            print("\n⚠️  Model download failed, but you can try again later.")
            print("You can manually run the test script to download models on first use.")
    
    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed. Please review the errors above.")
        return
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("You can now run:")
    print("  python test_indictrans.py")
    print("\nFor more advanced usage, check the README.md file.")

if __name__ == "__main__":
    main() 