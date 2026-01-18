# LiveKit Agent Dockerfile
# CPU Only - GPU operations delegated to AI Server

FROM python:3.12-slim-bookworm

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install system deps (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Download turn detector model files
RUN python -m agent.main download-files

ENV PYTHONUNBUFFERED=1

# Default to main.py with start command
CMD ["python", "-m", "agent.main", "start"]
