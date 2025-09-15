#!/usr/bin/env python3
"""
Script để pre-download models cho FaceFusion
Sử dụng trong Docker build process
"""

import os
import sys
sys.path.append('/app')

from facefusion.download import conditional_download_sources, conditional_download_hashes, resolve_download_url
from facefusion.filesystem import resolve_relative_path

def download_essential_models():
    """Download các models cần thiết cho face_enhancer và face_swapper"""
    
    # Essential models configuration
    models_config = {
        # Face Enhancer Models
        'codeformer': {
            'type': 'face_enhancer',
            'hash_file': 'codeformer.hash',
            'model_file': 'codeformer.onnx'
        },
        'gfpgan_1.4': {
            'type': 'face_enhancer', 
            'hash_file': 'gfpgan_1.4.hash',
            'model_file': 'gfpgan_1.4.onnx'
        },
        'gpen_bfr_512': {
            'type': 'face_enhancer',
            'hash_file': 'gpen_bfr_512.hash', 
            'model_file': 'gpen_bfr_512.onnx'
        },
        'restoreformer_plus_plus': {
            'type': 'face_enhancer',
            'hash_file': 'restoreformer_plus_plus.hash',
            'model_file': 'restoreformer_plus_plus.onnx'
        },
        # Face Swapper Models
        'inswapper_128_fp16': {
            'type': 'face_swapper',
            'hash_file': 'inswapper_128_fp16.hash',
            'model_file': 'inswapper_128_fp16.onnx'
        },
        'blendswap_256': {
            'type': 'face_swapper',
            'hash_file': 'blendswap_256.hash',
            'model_file': 'blendswap_256.onnx'  
        }
    }
    
    # Common models (face detection, etc)
    common_models = [
        'retinaface.hash', 'retinaface.onnx',
        'scrfd.hash', 'scrfd.onnx',
        'yoloface.hash', 'yoloface.onnx'
    ]
    
    total_downloaded = 0
    total_failed = 0
    
    print("🚀 Starting essential models download...")
    
    # Download essential models
    for model_name, config in models_config.items():
        print(f"\n📥 Downloading {model_name} ({config['type']})...")
        
        try:
            # Build paths
            hash_path = resolve_relative_path(f"../.assets/models/{config['hash_file']}")
            model_path = resolve_relative_path(f"../.assets/models/{config['model_file']}")
            
            # Download hash
            hash_set = {
                'hash': {
                    'url': resolve_download_url('models-3.0.0', config['hash_file']),
                    'path': hash_path
                }
            }
            
            # Download model
            source_set = {
                'source': {
                    'url': resolve_download_url('models-3.0.0', config['model_file']),
                    'path': model_path
                }
            }
            
            # Execute downloads
            if conditional_download_hashes(hash_set) and conditional_download_sources(source_set):
                # Verify file exists and has reasonable size
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1024 * 1024:  # > 1MB
                    size_mb = os.path.getsize(model_path) / 1024 / 1024
                    print(f"✅ {model_name} downloaded successfully ({size_mb:.1f}MB)")
                    total_downloaded += 1
                else:
                    print(f"❌ {model_name} download failed - file too small or missing")
                    total_failed += 1
            else:
                print(f"❌ {model_name} download failed")
                total_failed += 1
                
        except Exception as e:
            print(f"❌ {model_name} download error: {e}")
            total_failed += 1
    
    # Download common models
    print(f"\n📥 Downloading common models...")
    for model_file in common_models:
        try:
            model_path = resolve_relative_path(f"../.assets/models/{model_file}")
            
            if model_file.endswith('.hash'):
                hash_set = {
                    'hash': {
                        'url': resolve_download_url('models-3.0.0', model_file),
                        'path': model_path
                    }
                }
                conditional_download_hashes(hash_set)
            else:
                source_set = {
                    'source': {
                        'url': resolve_download_url('models-3.0.0', model_file), 
                        'path': model_path
                    }
                }
                conditional_download_sources(source_set)
                
            if os.path.exists(model_path):
                print(f"✅ {model_file} downloaded")
                
        except Exception as e:
            print(f"⚠️ {model_file} download failed: {e}")
    
    print(f"\n🎉 Download Summary:")
    print(f"✅ Success: {total_downloaded}")
    print(f"❌ Failed: {total_failed}")
    
    return total_failed == 0

if __name__ == "__main__":
    success = download_essential_models()
    sys.exit(0 if success else 1)
