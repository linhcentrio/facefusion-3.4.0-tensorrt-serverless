# Dockerfile cho FaceFusion RunPod Serverless với Pre-downloaded Models
FROM facefusion/facefusion:3.4.0-tensorrt

# Set working directory
WORKDIR /app

# Install additional Python packages for RunPod
RUN pip install --no-cache-dir \
    runpod \
    minio \
    psutil \
    requests \
    urllib3>=1.26.0

# Copy handler file
COPY facefusion_handler.py /app/

# Create models directory
RUN mkdir -p /app/.assets/models

# Pre-download essential models using FaceFusion force-download command
RUN python /app/facefusion.py --processors face_enhancer face_swapper force-download || echo "Some models may have failed to download, continuing..."

# Verify downloaded models (Fixed multiline Python)
RUN python -c "\
import os; \
models_dir = '/app/.assets/models'; \
onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')] if os.path.exists(models_dir) else []; \
print(f'Pre-downloaded {len(onnx_models)} models: {onnx_models[:5]}...' if len(onnx_models) > 5 else f'Pre-downloaded models: {onnx_models}'); \
"

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create required directories
RUN mkdir -p /app/temp /app/inputs /app/outputs

# Set permissions
RUN chmod +x /app/facefusion_handler.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "from facefusion_handler import health_check; health_ok, msg, info = health_check(); exit(0 if health_ok else 1)"

# Start the handler
CMD ["python", "-u", "/app/facefusion_handler.py"]
