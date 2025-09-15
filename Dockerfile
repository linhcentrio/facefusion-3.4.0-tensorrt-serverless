# Dockerfile tối ưu với pre-download models
FROM facefusion/facefusion:3.4.0-tensorrt

WORKDIR /app

# Install RunPod dependencies
RUN pip install --no-cache-dir runpod minio psutil requests urllib3>=1.26.0

# Copy pre-download script
COPY pre_download_models.py /app/
COPY facefusion_handler.py /app/

# Pre-download essential models
RUN python /app/pre_download_models.py

# Verify models
RUN python -c "
import os
models_dir = '/app/.assets/models'
onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')] if os.path.exists(models_dir) else []
print(f'Pre-downloaded {len(onnx_models)} models: {onnx_models}')
"

# Environment setup
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create directories
RUN mkdir -p /app/temp /app/inputs /app/outputs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from facefusion_handler import health_check; health_ok, msg, info = health_check(); exit(0 if health_ok else 1)"

CMD ["python", "-u", "/app/facefusion_handler.py"]
