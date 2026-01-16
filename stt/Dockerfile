FROM fedirz/faster-whisper-server:latest-cuda

USER root

# 1. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 모델 미리 다운로드
# /bin/uv에 설치되어 있으므로 그냥 uv라고 실행하면 됩니다.
WORKDIR /root/faster-whisper-server
RUN uv run python -c "from faster_whisper import WhisperModel; \
    WhisperModel('deepdml/faster-whisper-large-v3-turbo-ct2', device='cpu')"

# 3. 환경 변수 설정
# 베이스 이미지 규격에 맞춰 언더바 2개(__)를 사용합니다.
ENV WHISPER__MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
ENV WHISPER__INFERENCE_DEVICE=cuda
ENV WHISPER__COMPUTE_TYPE=float16

EXPOSE 8000

# 4. 서버 실행
CMD ["uv", "run", "uvicorn", "--factory", "faster_whisper_server.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
