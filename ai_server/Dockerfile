# AI Server Dockerfile
# GPU: CUDA 12.4 + cuDNN + TensorRT + Python 3.12
# Services: Whisper STT + Supertonic-2 TTS + BGE-M3 Embeddings

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install Python 3.12, ffmpeg, git-lfs, and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ffmpeg \
    libsndfile1 \
    git \
    git-lfs \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && git lfs install

WORKDIR /app

# Create TensorRT engine cache directory
RUN mkdir -p /opt/models/trt_cache

# Install dependencies (cache layer)
# NOTE: Other packages install onnxruntime (CPU) as dependency
# We install requirements first, then replace CPU with GPU version
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime onnxruntime-gpu || true \
    && pip install --no-cache-dir onnxruntime-gpu==1.20.1

# Copy application code
COPY . .

# Download Supertonic-2 TTS model
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Supertone/supertonic-2', local_dir='/opt/models/supertonic-2', ignore_patterns=['*.md', '*.gitattributes', 'img/*'])"

# Download pre-converted BGE-M3 ONNX model (no optimum dependency needed)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('aapot/bge-m3-onnx', local_dir='/opt/models/bge-m3-onnx', ignore_patterns=['*.md', '*.txt', '.gitattributes'])"

ENV PYTHONUNBUFFERED=1
ENV TTS_MODEL_PATH=/opt/models/supertonic-2
ENV BGE_M3_MODEL_PATH=/opt/models/bge-m3-onnx
ENV TRT_CACHE_DIR=/opt/models/trt_cache

# cuDNN library path for onnxruntime-gpu
# pip-installed nvidia-cudnn-cu12 is not in default LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# Expose AI Server port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run AI Server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
