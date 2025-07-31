#!/usr/bin/env python3
"""
Utility script to list all available pretrained models in the organized folder structure.
"""

import os
import glob
from pathlib import Path

def list_pretrained_models():
    """List all pretrained models in the organized folder structure"""
    
    print("🔍 PINNLab Pretrained Models Directory Structure")
    print("=" * 60)
    
    pretrained_dir = Path("pretrained_models")
    
    if not pretrained_dir.exists():
        print("❌ pretrained_models directory not found!")
        return
    
    # List all equation types
    for equation_dir in sorted(pretrained_dir.iterdir()):
        if equation_dir.is_dir():
            equation_name = equation_dir.name
            print(f"\n📁 {equation_name.upper()} MODELS")
            print("-" * 40)
            
            # Find all .pth files in this directory
            model_files = list(equation_dir.glob("*.pth"))
            
            if not model_files:
                print("   (No models found)")
                continue
            
            for model_file in sorted(model_files):
                file_size = model_file.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                print(f"   📄 {model_file.name} ({file_size_mb:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    
    # Count models by type
    total_models = 0
    for equation_dir in sorted(pretrained_dir.iterdir()):
        if equation_dir.is_dir():
            model_count = len(list(equation_dir.glob("*.pth")))
            if model_count > 0:
                print(f"   {equation_dir.name}: {model_count} models")
                total_models += model_count
    
    print(f"   Total: {total_models} models")
    
    print("\n💡 Usage:")
    print("   - Models are automatically organized by equation type")
    print("   - New models are saved in the appropriate subfolder")
    print("   - Web interface automatically detects available models")
    print("   - Use the web interface to visualize these models")

if __name__ == "__main__":
    list_pretrained_models() 