#!/usr/bin/env python3
"""Pre-download essential models for FaceFusion"""

import os
import sys
import subprocess

def main():
    print("🚀 Starting model pre-download...")
    
    # Method 1: Use FaceFusion built-in force-download
    try:
        print("📥 Using FaceFusion force-download...")
        result = subprocess.run([
            'python', '/app/facefusion.py',
            '--processors', 'face_enhancer', 'face_swapper',
            'force-download'
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode == 0:
            print("✅ Force-download completed successfully")
        else:
            print(f"⚠️ Force-download finished with warnings: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠️ Force-download timed out, but some models may have been downloaded")
    except Exception as e:
        print(f"❌ Force-download failed: {e}")
    
    # Verify downloaded models
    models_dir = '/app/.assets/models'
    if os.path.exists(models_dir):
        onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
        hash_files = [f for f in os.listdir(models_dir) if f.endswith('.hash')]
        
        print(f"📊 Found {len(onnx_models)} ONNX models and {len(hash_files)} hash files")
        
        # List key models
        essential_models = ['codeformer.onnx', 'gfpgan_1.4.onnx', 'inswapper_128_fp16.onnx']
        for model in essential_models:
            if model in onnx_models:
                size_mb = os.path.getsize(os.path.join(models_dir, model)) / 1024 / 1024
                print(f"✅ {model} ({size_mb:.1f}MB)")
            else:
                print(f"❌ {model} not found")
    else:
        print("❌ Models directory not found")
    
    print("🎉 Pre-download process completed!")

if __name__ == "__main__":
    main()
