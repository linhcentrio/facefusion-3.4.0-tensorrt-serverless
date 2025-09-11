# FaceFusion RunPod Serverless Service

Dịch vụ FaceFusion được tối ưu hóa cho RunPod Serverless với hỗ trợ GPU TensorRT, cung cấp khả năng Face Enhancement, Face Swapping và Frame Enhancement chất lượng cao.

## 🚀 Tính năng

- **Face Enhancement**: Cải thiện chất lượng khuôn mặt với các model CodeFormer, GFPGAN, GPEN
- **Face Swapping**: Hoán đổi khuôn mặt với model InSwapper, SimSwap
- **Frame Enhancement**: Nâng cấp chất lượng khung hình với Real-ESRGAN
- **TensorRT Optimization**: Hỗ trợ TensorRT, CUDA và CPU với auto-detection tối ưu
- **Robust Download**: Hệ thống download với fallback (requests → wget → curl)
- **MinIO Integration**: Upload kết quả tự động lên cloud storage

## 📦 Cấu trúc Files

```
├── facefusion_handler.py          # Handler chính cho RunPod
├── Dockerfile.facefusion           # Dockerfile với base image TensorRT
├── docker-compose.facefusion.yml   # Docker compose cho local testing
├── test_facefusion_service.py      # Script test service
└── README_FaceFusion.md           # Hướng dẫn này
```

## 🛠️ Cài đặt

### 1. Build Docker Image

```bash
# Build image
docker build -f Dockerfile.facefusion -t facefusion-runpod .

# Hoặc sử dụng docker-compose
docker-compose -f docker-compose.facefusion.yml build
```

### 2. Environment Variables

Cấu hình các biến môi trường cần thiết:

```bash
# MinIO Storage
export MINIO_ENDPOINT="media.aiclip.ai"
export MINIO_ACCESS_KEY="your_access_key"
export MINIO_SECRET_KEY="your_secret_key"
export MINIO_BUCKET="video"
export MINIO_SECURE="false"

# GPU Settings
export NVIDIA_VISIBLE_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"
```

## 📋 Sử dụng

### 1. Face Enhancement Only

```json
{
  "input": {
    "target_url": "https://example.com/video.mp4",
    "frame_processors": ["face_enhancer"],
    "face_enhancer_model": "codeformer",
    "face_enhancer_blend": 80,
    "face_detector_model": "retinaface",
    "face_detector_size": "640x640",
    "execution_providers": ["cuda"],
    "output_video_encoder": "libx264",
    "keep_fps": true,
    "skip_audio": false
  }
}
```

### 2. Face Swapping Only

```json
{
  "input": {
    "source_url": "https://example.com/source_face.jpg",
    "target_url": "https://example.com/target_video.mp4",
    "frame_processors": ["face_swapper"],
    "face_swapper_model": "inswapper_128_fp16",
    "face_detector_model": "retinaface",
    "face_selector_mode": "one",
    "reference_face_position": 0,
    "execution_providers": ["cuda"]
  }
}
```

### 3. Combined Processing (Face Swap + Enhancement)

```json
{
  "input": {
    "source_url": "https://example.com/source_face.jpg",
    "target_url": "https://example.com/target_video.mp4",
    "frame_processors": ["face_swapper", "face_enhancer"],
    "face_swapper_model": "inswapper_128_fp16",
    "face_enhancer_model": "codeformer",
    "face_enhancer_blend": 80,
    "face_detector_model": "retinaface",
    "face_detector_size": "640x640",
    "execution_providers": ["tensorrt"],
    "execution_thread_count": 8,
    "execution_queue_count": 2
  }
}
```

### 4. TensorRT Optimization (Khuyến nghị)

```json
{
  "input": {
    "target_url": "https://example.com/video.mp4",
    "frame_processors": ["face_enhancer"],
    "face_enhancer_model": "codeformer",
    "execution_providers": ["tensorrt"],
    "execution_thread_count": 16,
    "execution_queue_count": 4
  }
}
```

**Lưu ý TensorRT:**
- Service tự động detect TensorRT availability
- TensorRT sẽ được ưu tiên nếu có sẵn
- Hiệu suất tăng 2-5x so với CUDA
- Chỉ hoạt động với base image TensorRT

## 🎛️ Tham số cấu hình

### Core Parameters

| Tham số | Loại | Mô tả | Mặc định |
|---------|------|--------|-----------|
| `target_url` | string | **Bắt buộc** - URL file video/ảnh đích | - |
| `source_url` | string | URL ảnh khuôn mặt nguồn (cho face swapping) | - |
| `frame_processors` | array | Các processor cần sử dụng | `["face_enhancer"]` |

### Face Enhancement

| Tham số | Giá trị | Mô tả |
|---------|---------|--------|
| `face_enhancer_model` | `codeformer`, `gfpgan_1_4`, `gpen_bfr_512` | Model enhancement |
| `face_enhancer_blend` | 0-100 | Mức độ blend với ảnh gốc |

### Face Swapping

| Tham số | Giá trị | Mô tả |
|---------|---------|--------|
| `face_swapper_model` | `inswapper_128`, `inswapper_128_fp16`, `simswap_256` | Model face swapping |
| `face_selector_mode` | `one`, `many`, `reference` | Chế độ chọn khuôn mặt |
| `reference_face_position` | 0-N | Vị trí khuôn mặt tham chiếu |

### Face Detection

| Tham số | Giá trị | Mô tả |
|---------|---------|--------|
| `face_detector_model` | `retinaface`, `scrfd`, `yoloface` | Model detect khuôn mặt |
| `face_detector_size` | `320x320`, `640x640`, `1024x1024` | Kích thước detection |
| `face_detector_score` | 0.0-1.0 | Ngưỡng confidence |

### Performance Settings

| Tham số | Loại | Mô tả | Mặc định |
|---------|------|--------|-----------|
| `execution_providers` | array | `["tensorrt"]`, `["cuda"]`, `["cpu"]` | Auto-detect (TensorRT > CUDA > CPU) |
| `execution_thread_count` | int | Số thread xử lý | 4 |
| `execution_queue_count` | int | Số queue xử lý | 1 |

### Output Settings

| Tham số | Loại | Mô tả | Mặc định |
|---------|------|--------|-----------|
| `output_video_encoder` | string | `libx264`, `libx265`, `h264_nvenc` | `libx264` |
| `output_video_quality` | int | Chất lượng video (0-100) | 100 |
| `output_image_quality` | int | Chất lượng ảnh (0-100) | 100 |
| `keep_fps` | boolean | Giữ nguyên FPS | `true` |
| `skip_audio` | boolean | Bỏ qua audio | `false` |

## 🧪 Testing

Chạy script test để kiểm tra service:

```bash
# Cập nhật endpoint URL trong test_facefusion_service.py
python test_facefusion_service.py
```

Hoặc test bằng curl:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "target_url": "https://example.com/video.mp4",
      "frame_processors": ["face_enhancer"],
      "face_enhancer_model": "codeformer"
    }
  }'
```

## 📊 Response Format

### Success Response

```json
{
  "output_url": "https://media.aiclip.ai/video/facefusion/...",
  "status": "completed",
  "job_id": "uuid-job-id",
  "processing_time_seconds": 45.67,
  "file_info": {
    "target_size_mb": 15.2,
    "source_size_mb": 2.1,
    "output_size_mb": 18.5,
    "type": "video"
  },
  "config": {
    "frame_processors": ["face_swapper", "face_enhancer"],
    "face_swapper_model": "inswapper_128_fp16",
    "face_enhancer_model": "codeformer"
  },
  "performance": {
    "download_time": 5.2,
    "processing_time": 40.47,
    "total_time": 45.67
  },
  "gpu_info": {
    "torch_cuda_available": true,
    "gpu_name": "NVIDIA RTX 4090",
    "onnx_gpu_available": true
  },
  "version": "1.0_FACEFUSION_OPTIMIZED"
}
```

### Error Response

```json
{
  "error": "Processing failed: Face not detected",
  "job_id": "uuid-job-id",
  "processing_time_seconds": 12.34,
  "debug_info": {
    "stdout": "...",
    "stderr": "..."
  }
}
```

## 🔧 Troubleshooting

### Lỗi thường gặp

1. **"Models missing"**
   - Base image chưa download đầy đủ models
   - Kiểm tra health check endpoint

2. **"GPU not available"**
   - Kiểm tra NVIDIA drivers
   - Đảm bảo `nvidia-container-runtime` được cài đặt

3. **"Download failed"**
   - URL không hợp lệ hoặc không accessible
   - Kiểm tra network connectivity

4. **"Processing timeout"**
   - File quá lớn (>30 phút xử lý)
   - Tăng timeout hoặc giảm chất lượng

### Health Check

```bash
# Kiểm tra health của service
curl http://localhost:8080/health

# Hoặc trong Docker
docker exec container_name python -c "from facefusion_handler import health_check; print(health_check())"
```

## 🚀 Deploy to RunPod

1. **Build và push image:**
```bash
docker build -f Dockerfile.facefusion -t your-registry/facefusion-runpod .
docker push your-registry/facefusion-runpod
```

2. **Tạo RunPod Serverless Endpoint:**
   - Image: `your-registry/facefusion-runpod`
   - GPU Type: RTX 4090, A100 (khuyến nghị)
   - Container Port: 8080
   - Environment Variables: Cấu hình MinIO

3. **Test endpoint:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @test_payload.json
```

## 📈 Performance Tips

### 🚀 TensorRT Optimization (Khuyến nghị cao)
- **Base Image**: Sử dụng `facefusion/facefusion:3.4.0-tensorrt`
- **Auto-Detection**: Service tự động detect và sử dụng TensorRT nếu có
- **Performance Gain**: 2-5x nhanh hơn CUDA
- **Explicit Usage**: Sử dụng `"execution_providers": ["tensorrt"]` trong input

### 🔧 Hardware & Configuration
- **GPU Memory**: RTX 4090 (24GB) khuyến nghị cho video chất lượng cao
- **Thread Count**: Tối ưu `execution_thread_count` dựa trên VRAM available
  - TensorRT: 16-32 threads
  - CUDA: 8-16 threads  
  - CPU: 4-8 threads
- **Batch Processing**: Sử dụng `execution_queue_count` > 1 cho multiple videos

### 🎯 Model Selection
- **Face Enhancement**: `codeformer` (tốt nhất cho TensorRT)
- **Face Swapping**: `inswapper_128_fp16` (tối ưu speed/quality)
- **Face Detection**: `retinaface` (accuracy cao nhất)

### ⚡ Execution Provider Priority
1. **TensorRT** - Nhanh nhất (nếu có sẵn)
2. **CUDA** - Tốt cho GPU NVIDIA
3. **CPU** - Fallback cho môi trường không có GPU

## 📝 License

Dự án sử dụng base image từ [FaceFusion](https://github.com/facefusion/facefusion) project.
