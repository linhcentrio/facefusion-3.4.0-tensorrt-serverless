#!/usr/bin/env python3
"""
RunPod Serverless Handler cho FaceFusion Service - GPU OPTIMIZED VERSION
Hỗ trợ Face Enhancement và Face Swapping với TensorRT optimization
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import sys
import json
import traceback
import subprocess
import psutil
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
import logging
from typing import Tuple, Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "media.aiclip.ai")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "VtZ6MUPfyTOH3qSiohA2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# FaceFusion base path
FACEFUSION_PATH = "/app"
FACEFUSION_SCRIPT = "/app/facefusion.py"

# Initialize MinIO
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    minio_client.bucket_exists(MINIO_BUCKET)
    logger.info(f"✅ MinIO connected: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
except Exception as e:
    logger.error(f"❌ MinIO failed: {e}")
    minio_client = None

# Supported processors and models
SUPPORTED_PROCESSORS = {
    'face_enhancer': {
        'models': ['codeformer', 'gfpgan_1_4', 'gpen_bfr_256', 'gpen_bfr_512', 'restoreformer'],
        'description': 'Enhance face quality'
    },
    'face_swapper': {
        'models': ['blendswap_256', 'inswapper_128', 'inswapper_128_fp16', 'simswap_256', 'simswap_512_unofficial'],
        'description': 'Swap faces between source and target'
    },
    'frame_enhancer': {
        'models': ['real_esrgan_x2plus', 'real_esrgan_x4plus', 'real_hatgan_x4'],
        'description': 'Enhance overall frame quality'
    }
}

FACE_DETECTOR_MODELS = ['retinaface', 'scrfd', 'yoloface']
FACE_DETECTOR_SIZES = ['320x320', '512x512', '640x640', '960x960', '1024x1024']

def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information"""
    try:
        import torch
        
        gpu_info = {
            'torch_cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            gpu_info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            })
            
        # Check ONNX Runtime GPU providers
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            gpu_providers = [p for p in available_providers if any(gpu in p for gpu in ['CUDA', 'GPU', 'DML', 'TensorRT'])]
            
            # Detect specific providers
            tensorrt_available = 'TensorrtExecutionProvider' in available_providers
            cuda_available = 'CUDAExecutionProvider' in available_providers
            
            gpu_info.update({
                'onnx_version': ort.__version__,
                'onnx_providers': available_providers,
                'onnx_gpu_providers': gpu_providers,
                'onnx_gpu_available': len(gpu_providers) > 0,
                'tensorrt_available': tensorrt_available,
                'cuda_available': cuda_available,
                'recommended_provider': 'tensorrt' if tensorrt_available else ('cuda' if cuda_available else 'cpu')
            })
            
        except Exception as e:
            logger.warning(f"ONNX Runtime info error: {e}")
            
        return gpu_info
        
    except Exception as e:
        logger.error(f"GPU info error: {e}")
        return {'error': str(e)}

def create_optimized_session() -> requests.Session:
    """Create requests session with fixed urllib3 compatibility"""
    session = requests.Session()
    
    # Fixed retry strategy - compatible with urllib3 1.26.x
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Fixed: was method_whitelist
        backoff_factor=2,
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        'User-Agent': 'FaceFusion-Service/1.0',
        'Accept': '*/*',
        'Connection': 'keep-alive'
    })
    
    return session

def download_file_robust(url: str, output_path: str, max_retries: int = 3) -> bool:
    """Enhanced download with multiple fallback methods"""
    
    # Method 1: Requests with fixed retry
    try:
        logger.info(f"📥 Downloading via requests: {url}")
        session = create_optimized_session()
        
        response = session.get(url, stream=True, timeout=(30, 300))
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
        if downloaded > 1024:  # At least 1KB
            logger.info(f"✅ Downloaded {downloaded/1024/1024:.1f}MB via requests")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ Requests download failed: {e}")

    # Method 2: wget fallback
    try:
        logger.info(f"📥 Downloading via wget: {url}")
        
        cmd = [
            'wget', '-q', '--tries=3', '--timeout=60',
            '--read-timeout=300', '-c', '--progress=bar:force',
            '-O', output_path, url
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"✅ Downloaded {size_mb:.1f}MB via wget")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ wget download failed: {e}")

    # Method 3: curl fallback
    try:
        logger.info(f"📥 Downloading via curl: {url}")
        
        cmd = [
            'curl', '-L', '-C', '-', '--max-time', '600',
            '--connect-timeout', '30', '--retry', '3',
            '--silent', '--show-error', '--fail',
            '-o', output_path, url
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=700)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"✅ Downloaded {size_mb:.1f}MB via curl")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ curl download failed: {e}")

    return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload with error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO not available")
            
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading {file_size:.1f}MB: {object_name}")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Uploaded: {url}")
        return url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

def get_file_type(file_path: str) -> str:
    """Detect file type"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm', '.m4v']
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if ext in video_exts:
            return 'video'
        elif ext in image_exts:
            return 'image'
        else:
            return 'unknown'
    except:
        return 'unknown'

def build_facefusion_command(config: Dict[str, Any], input_path: str, source_path: str, output_path: str) -> List[str]:
    """Build FaceFusion CLI command based on configuration"""
    
    cmd = ['python', FACEFUSION_SCRIPT, '--headless']
    
    # Execution providers
    execution_providers = config.get('execution_providers', ['cuda'])
    if isinstance(execution_providers, str):
        execution_providers = [execution_providers]
    cmd.extend(['--execution-providers'] + execution_providers)
    
    # Performance settings
    if config.get('execution_thread_count'):
        cmd.extend(['--execution-thread-count', str(config['execution_thread_count'])])
    if config.get('execution_queue_count'):
        cmd.extend(['--execution-queue-count', str(config['execution_queue_count'])])
    
    # Frame processors
    processors = config.get('frame_processors', ['face_enhancer'])
    if isinstance(processors, str):
        processors = [processors]
    cmd.extend(['--frame-processors'] + processors)
    
    # Source and target
    if source_path:
        cmd.extend(['-s', source_path])
    cmd.extend(['-t', input_path])
    cmd.extend(['-o', output_path])
    
    # Face enhancer settings
    if 'face_enhancer' in processors:
        if config.get('face_enhancer_model'):
            cmd.extend(['--face-enhancer-model', config['face_enhancer_model']])
        if config.get('face_enhancer_blend') is not None:
            cmd.extend(['--face-enhancer-blend', str(config['face_enhancer_blend'])])
    
    # Face swapper settings
    if 'face_swapper' in processors:
        if config.get('face_swapper_model'):
            cmd.extend(['--face-swapper-model', config['face_swapper_model']])
    
    # Frame enhancer settings
    if 'frame_enhancer' in processors:
        if config.get('frame_enhancer_model'):
            cmd.extend(['--frame-enhancer-model', config['frame_enhancer_model']])
        if config.get('frame_enhancer_blend') is not None:
            cmd.extend(['--frame-enhancer-blend', str(config['frame_enhancer_blend'])])
    
    # Face detection settings
    if config.get('face_detector_model'):
        cmd.extend(['--face-detector-model', config['face_detector_model']])
    if config.get('face_detector_size'):
        cmd.extend(['--face-detector-size', config['face_detector_size']])
    if config.get('face_detector_score') is not None:
        cmd.extend(['--face-detector-score', str(config['face_detector_score'])])
    
    # Face selection settings
    if config.get('face_selector_mode'):
        cmd.extend(['--face-selector-mode', config['face_selector_mode']])
    if config.get('face_analyser_order'):
        cmd.extend(['--face-analyser-order', config['face_analyser_order']])
    if config.get('face_analyser_age'):
        cmd.extend(['--face-analyser-age', config['face_analyser_age']])
    if config.get('face_analyser_gender'):
        cmd.extend(['--face-analyser-gender', config['face_analyser_gender']])
    
    # Reference settings
    if config.get('reference_face_position') is not None:
        cmd.extend(['--reference-face-position', str(config['reference_face_position'])])
    if config.get('reference_frame_number') is not None:
        cmd.extend(['--reference-frame-number', str(config['reference_frame_number'])])
    
    # Frame processing settings
    if config.get('temp_frame_format'):
        cmd.extend(['--temp-frame-format', config['temp_frame_format']])
    if config.get('temp_frame_quality') is not None:
        cmd.extend(['--temp-frame-quality', str(config['temp_frame_quality'])])
    
    # Output settings
    if config.get('output_image_quality') is not None:
        cmd.extend(['--output-image-quality', str(config['output_image_quality'])])
    if config.get('output_video_encoder'):
        cmd.extend(['--output-video-encoder', config['output_video_encoder']])
    if config.get('output_video_quality') is not None:
        cmd.extend(['--output-video-quality', str(config['output_video_quality'])])
    
    # Flags
    if config.get('keep_temp', False):
        cmd.append('--keep-temp')
    if config.get('keep_fps', True):
        cmd.append('--keep-fps')
    if config.get('skip_audio', False):
        cmd.append('--skip-audio')
    
    return cmd

def run_facefusion_processing(input_path: str, source_path: Optional[str], output_path: str, 
                             config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Run FaceFusion processing with given configuration"""
    try:
        logger.info(f"🎨 Starting FaceFusion processing")
        
        # Build command
        cmd = build_facefusion_command(config, input_path, source_path, output_path)
        
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        
        start_time = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=FACEFUSION_PATH
        )
        
        # Real-time monitoring
        while process.poll() is None:
            # Timeout check (30 minutes)
            if time.time() - start_time > 1800:
                process.terminate()
                return False, "Timeout (30 min limit)", {}
                
            time.sleep(2)
            
        stdout, stderr = process.communicate()
        processing_time = time.time() - start_time
        
        stats = {
            'processing_time': round(processing_time, 2),
            'return_code': process.returncode,
            'stdout': stdout[-2000:] if stdout else "",  # Last 2000 chars
            'stderr': stderr[-2000:] if stderr else ""   # Last 2000 chars
        }
        
        if process.returncode == 0 and os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            stats['output_size_mb'] = round(output_size, 2)
            
            logger.info(f"✅ Processing completed in {processing_time:.1f}s → {output_size:.1f}MB")
            return True, f"Success in {processing_time:.1f}s", stats
        else:
            error_msg = stderr.strip() if stderr else "Unknown error"
            logger.error(f"❌ Processing failed: {error_msg}")
            return False, error_msg, stats
            
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        return False, str(e), {}

def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate configuration parameters"""
    
    # Check required fields
    processors = config.get('frame_processors', [])
    if not processors:
        return False, "frame_processors is required"
    
    # Validate processors
    for processor in processors:
        if processor not in SUPPORTED_PROCESSORS:
            return False, f"Unsupported processor: {processor}"
    
    # Validate models
    for processor in processors:
        model_key = f"{processor}_model"
        if model_key in config:
            model = config[model_key]
            if model not in SUPPORTED_PROCESSORS[processor]['models']:
                return False, f"Unsupported {processor} model: {model}"
    
    # Validate face detector
    if 'face_detector_model' in config:
        if config['face_detector_model'] not in FACE_DETECTOR_MODELS:
            return False, f"Unsupported face detector model: {config['face_detector_model']}"
    
    if 'face_detector_size' in config:
        if config['face_detector_size'] not in FACE_DETECTOR_SIZES:
            return False, f"Unsupported face detector size: {config['face_detector_size']}"
    
    return True, "Valid configuration"

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler for FaceFusion processing"""
    job_id = job.get("id", str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Job {job_id}: Starting FaceFusion processing")
        
        # Get GPU status
        gpu_info = get_gpu_info()
        logger.info(f"🖥️ GPU Status: CUDA: {gpu_info.get('torch_cuda_available', False)} | ONNX GPU: {gpu_info.get('onnx_gpu_available', False)} | TensorRT: {gpu_info.get('tensorrt_available', False)}")
        logger.info(f"🚀 Recommended Provider: {gpu_info.get('recommended_provider', 'cpu')}")
        
        job_input = job.get("input", {})
        
        # Validate required inputs
        if not job_input.get("target_url"):
            return {"error": "target_url is required", "job_id": job_id}
        
        target_url = job_input["target_url"]
        source_url = job_input.get("source_url")  # Optional for face enhancement only
        
        # Configuration with defaults - prioritize TensorRT if available
        default_provider = gpu_info.get('recommended_provider', 'cpu')
        config = {
            'frame_processors': job_input.get('frame_processors', ['face_enhancer']),
            'execution_providers': job_input.get('execution_providers', [default_provider]),
            'face_enhancer_model': job_input.get('face_enhancer_model', 'codeformer'),
            'face_enhancer_blend': job_input.get('face_enhancer_blend', 80),
            'face_swapper_model': job_input.get('face_swapper_model', 'inswapper_128_fp16'),
            'face_detector_model': job_input.get('face_detector_model', 'retinaface'),
            'face_detector_size': job_input.get('face_detector_size', '640x640'),
            'face_detector_score': job_input.get('face_detector_score', 0.5),
            'face_selector_mode': job_input.get('face_selector_mode', 'one'),
            'temp_frame_format': job_input.get('temp_frame_format', 'jpg'),
            'temp_frame_quality': job_input.get('temp_frame_quality', 100),
            'output_image_quality': job_input.get('output_image_quality', 100),
            'output_video_encoder': job_input.get('output_video_encoder', 'libx264'),
            'output_video_quality': job_input.get('output_video_quality', 100),
            'keep_fps': job_input.get('keep_fps', True),
            'skip_audio': job_input.get('skip_audio', False),
            'keep_temp': job_input.get('keep_temp', False)
        }
        
        # Update config with additional parameters
        for key, value in job_input.items():
            if key not in config and not key.endswith('_url'):
                config[key] = value
        
        # Validate configuration
        is_valid, validation_msg = validate_config(config)
        if not is_valid:
            return {"error": f"Invalid configuration: {validation_msg}", "job_id": job_id}
        
        logger.info(f"🎨 Config: {config['frame_processors']} with {config.get('execution_providers', ['cpu'])}")
        
        # Process files
        with tempfile.TemporaryDirectory(prefix=f"facefusion_{job_id}_") as temp_dir:
            # Download target file
            parsed_target = urlparse(target_url)
            target_filename = os.path.basename(parsed_target.path) or f"target_{int(time.time())}.mp4"
            target_path = os.path.join(temp_dir, target_filename)
            
            logger.info("📥 Downloading target file...")
            download_start = time.time()
            
            if not download_file_robust(target_url, target_path):
                return {"error": "Target download failed", "job_id": job_id}
            
            download_time = time.time() - download_start
            
            if not os.path.exists(target_path) or os.path.getsize(target_path) < 1024:
                return {"error": "Downloaded target file invalid", "job_id": job_id}
            
            target_size = os.path.getsize(target_path) / (1024 * 1024)
            file_type = get_file_type(target_path)
            
            logger.info(f"📊 Target: {file_type}, {target_size:.1f}MB, downloaded in {download_time:.1f}s")
            
            # Download source file if provided
            source_path = None
            source_size = 0
            if source_url:
                parsed_source = urlparse(source_url)
                source_filename = os.path.basename(parsed_source.path) or f"source_{int(time.time())}.jpg"
                source_path = os.path.join(temp_dir, source_filename)
                
                logger.info("📥 Downloading source file...")
                if not download_file_robust(source_url, source_path):
                    return {"error": "Source download failed", "job_id": job_id}
                
                if not os.path.exists(source_path) or os.path.getsize(source_path) < 1024:
                    return {"error": "Downloaded source file invalid", "job_id": job_id}
                
                source_size = os.path.getsize(source_path) / (1024 * 1024)
                logger.info(f"📊 Source: {source_size:.1f}MB")
            
            # Generate output filename
            name, ext = os.path.splitext(target_filename)
            processors_str = "_".join(config['frame_processors'])
            output_filename = f"{name}_{processors_str}_processed{'.mp4' if file_type == 'video' else ext}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Process
            logger.info("🎨 Starting FaceFusion processing...")
            success, message, stats = run_facefusion_processing(
                target_path, source_path, output_path, config
            )
            
            if not success:
                return {
                    "error": f"Processing failed: {message}", 
                    "job_id": job_id, 
                    "stats": stats,
                    "debug_info": {
                        "stdout": stats.get('stdout', ''),
                        "stderr": stats.get('stderr', '')
                    }
                }
            
            # Upload result
            logger.info("📤 Uploading result...")
            timestamp = int(time.time())
            object_name = f"facefusion/{processors_str}/{timestamp}/{job_id}_{output_filename}"
            
            try:
                output_url = upload_to_minio(output_path, object_name)
            except Exception as e:
                return {"error": f"Upload failed: {e}", "job_id": job_id}
            
            # Success response
            total_time = time.time() - start_time
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            
            logger.info(f"✅ Job {job_id} completed in {total_time:.1f}s")
            
            return {
                "output_url": output_url,
                "status": "completed",
                "job_id": job_id,
                "processing_time_seconds": round(total_time, 2),
                "file_info": {
                    "target_size_mb": round(target_size, 2),
                    "source_size_mb": round(source_size, 2) if source_path else None,
                    "output_size_mb": round(output_size, 2),
                    "type": file_type
                },
                "config": config,
                "performance": {
                    "download_time": round(download_time, 2),
                    "processing_time": stats.get('processing_time', 0),
                    "total_time": round(total_time, 2)
                },
                "gpu_info": gpu_info,
                "version": "1.0_FACEFUSION_OPTIMIZED"
            }
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Internal error: {str(e)}",
            "job_id": job_id,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

def health_check() -> Tuple[bool, str, Dict[str, Any]]:
    """Health check for FaceFusion service"""
    try:
        # Check GPU
        gpu_info = get_gpu_info()
        
        # Check FaceFusion script
        facefusion_ok = os.path.exists(FACEFUSION_SCRIPT)
        
        # Check MinIO
        minio_ok = minio_client is not None
        
        health_info = {
            "gpu_info": gpu_info,
            "facefusion": {"script_available": facefusion_ok, "path": FACEFUSION_SCRIPT},
            "storage": {"minio_available": minio_ok},
            "supported_processors": SUPPORTED_PROCESSORS,
            "status": "healthy" if (facefusion_ok and minio_ok) else "degraded"
        }
        
        if facefusion_ok and minio_ok:
            logger.info("✅ Health check passed")
            return True, "All systems operational", health_info
        else:
            issues = []
            if not facefusion_ok:
                issues.append("FaceFusion script missing")
            if not minio_ok:
                issues.append("MinIO unavailable")
            return False, "; ".join(issues), health_info
            
    except Exception as e:
        return False, f"Health check error: {e}", {"error": str(e)}

# Startup sequence
if __name__ == "__main__":
    logger.info("🚀 FaceFusion Service - GPU Optimized v1.0")
    
    try:
        # Verify GPU setup
        gpu_info = get_gpu_info()
        logger.info(f"🔥 GPU Status:")
        logger.info(f"   PyTorch CUDA: {gpu_info.get('torch_cuda_available', False)}")
        logger.info(f"   ONNX GPU: {gpu_info.get('onnx_gpu_available', False)}")
        logger.info(f"   TensorRT: {gpu_info.get('tensorrt_available', False)}")
        logger.info(f"   Recommended Provider: {gpu_info.get('recommended_provider', 'cpu')}")
        if gpu_info.get('gpu_name'):
            logger.info(f"   GPU: {gpu_info['gpu_name']}")
            
        # Health check
        health_ok, health_msg, health_info = health_check()
        if not health_ok:
            logger.warning(f"⚠️ Health check issues: {health_msg}")
        else:
            logger.info(f"✅ {health_msg}")
            
        logger.info(f"🎨 Ready: {len(SUPPORTED_PROCESSORS)} processors available")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)
