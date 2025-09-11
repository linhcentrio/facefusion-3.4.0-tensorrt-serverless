#!/usr/bin/env python3
"""
Script test để demo FaceFusion RunPod Service
"""

import requests
import json
import time
from typing import Dict, Any

def test_face_enhancement():
    """Test face enhancement only"""
    print("🎨 Testing Face Enhancement...")
    
    payload = {
        "input": {
            "target_url": "https://example.com/video.mp4",  # URL video cần enhance
            "frame_processors": ["face_enhancer"],
            "face_enhancer_model": "codeformer",
            "face_enhancer_blend": 80,
            "face_detector_model": "retinaface",
            "face_detector_size": "640x640",
            "execution_providers": ["tensorrt"],
            "output_video_encoder": "libx264",
            "output_video_quality": 100,
            "keep_fps": True,
            "skip_audio": False
        }
    }
    
    return payload

def test_face_swapping():
    """Test face swapping"""
    print("🔄 Testing Face Swapping...")
    
    payload = {
        "input": {
            "source_url": "https://example.com/source_face.jpg",  # URL ảnh khuôn mặt nguồn
            "target_url": "https://example.com/target_video.mp4",  # URL video đích
            "frame_processors": ["face_swapper"],
            "face_swapper_model": "inswapper_128_fp16",
            "face_detector_model": "retinaface",
            "face_detector_size": "640x640",
            "face_selector_mode": "one",
            "reference_face_position": 0,
            "reference_frame_number": 0,
            "execution_providers": ["tensorrt"],
            "output_video_encoder": "libx264",
            "output_video_quality": 100,
            "keep_fps": True,
            "skip_audio": False
        }
    }
    
    return payload

def test_combined_processing():
    """Test face swapping + enhancement"""
    print("🚀 Testing Combined Face Swapping + Enhancement...")
    
    payload = {
        "input": {
            "source_url": "https://example.com/source_face.jpg",  # URL ảnh khuôn mặt nguồn
            "target_url": "https://example.com/target_video.mp4",  # URL video đích
            "frame_processors": ["face_swapper", "face_enhancer"],
            
            # Face swapper settings
            "face_swapper_model": "inswapper_128_fp16",
            
            # Face enhancer settings  
            "face_enhancer_model": "codeformer",
            "face_enhancer_blend": 80,
            
            # Face detector settings
            "face_detector_model": "retinaface",
            "face_detector_size": "640x640",
            "face_detector_score": 0.5,
            
            # Face selector settings
            "face_selector_mode": "one",
            "face_analyser_order": "left-right",
            "face_analyser_age": "adult",
            "face_analyser_gender": "female",
            
            # Reference settings
            "reference_face_position": 0,
            "reference_frame_number": 0,
            
            # Performance settings
            "execution_providers": ["tensorrt"],
            "execution_thread_count": 16,
            "execution_queue_count": 2,
            
            # Frame processing
            "temp_frame_format": "jpg",
            "temp_frame_quality": 100,
            
            # Output settings
            "output_video_encoder": "libx264",
            "output_video_quality": 100,
            "keep_fps": True,
            "skip_audio": False,
            "keep_temp": False
        }
    }
    
    return payload

def send_request(endpoint_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Gửi request tới RunPod endpoint"""
    try:
        print(f"📤 Sending request to: {endpoint_url}")
        print(f"📝 Payload: {json.dumps(payload, indent=2)}")
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer YOUR_RUNPOD_API_KEY'  # Thay thế bằng API key thật
        }
        
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {"error": str(e)}

def monitor_job(job_id: str, endpoint_url: str) -> Dict[str, Any]:
    """Monitor job status (if RunPod supports job monitoring)"""
    print(f"🔍 Monitoring job: {job_id}")
    
    # Thực hiện logic monitor job nếu cần
    # Đây là placeholder - thực tế có thể cần endpoint khác để check status
    
    return {"status": "monitoring", "job_id": job_id}

def main():
    """Main test function"""
    print("🚀 FaceFusion Service Test Suite")
    print("=" * 50)
    
    # Cấu hình endpoint
    RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"  # Thay thế bằng endpoint thật
    
    # Test cases
    test_cases = [
        ("Face Enhancement", test_face_enhancement()),
        ("Face Swapping", test_face_swapping()),
        ("Combined Processing", test_combined_processing())
    ]
    
    results = []
    
    for test_name, payload in test_cases:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        result = send_request(RUNPOD_ENDPOINT, payload)
        end_time = time.time()
        
        result["test_name"] = test_name
        result["request_time"] = round(end_time - start_time, 2)
        results.append(result)
        
        # Thêm delay giữa các test
        time.sleep(2)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    for result in results:
        test_name = result.get("test_name", "Unknown")
        status = "✅ SUCCESS" if "output_url" in result else "❌ FAILED"
        request_time = result.get("request_time", 0)
        processing_time = result.get("processing_time_seconds", 0)
        
        print(f"{test_name}: {status}")
        print(f"  Request Time: {request_time}s")
        print(f"  Processing Time: {processing_time}s")
        
        if "output_url" in result:
            print(f"  Output URL: {result['output_url']}")
        elif "error" in result:
            print(f"  Error: {result['error']}")
        print()

if __name__ == "__main__":
    main()
