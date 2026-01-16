FROM python:3.12-slim

# Install minimal system deps (git-lfs for model download)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    git \
    git-lfs \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download turn detector model files
RUN python agent/agent.py download-files

# Download Supertonic-2 TTS model from Hugging Face (~263MB)
# /opt/models를 사용하여 볼륨 마운트에 영향받지 않도록 함
RUN mkdir -p /opt/models && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Supertone/supertonic-2', local_dir='/opt/models/supertonic-2', local_dir_use_symlinks=False, ignore_patterns=['*.md', '*.gitattributes', 'img/*'])" && \
    ls -la /opt/models/supertonic-2/

ENV PYTHONUNBUFFERED=1

# Default to agent.py with start command
CMD ["python", "agent/agent.py", "start"]
