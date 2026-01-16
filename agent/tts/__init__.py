"""Custom TTS plugins for ops-agent."""

from .custom_tts import CustomTTS

# Supertonic-2 ONNX TTS (다국어 지원: ko, en, es, pt, fr)
# 모델은 Docker 빌드 시 /app/models/supertonic-2에 다운로드됨
try:
    from .supertonic_tts import SupertonicTTS
except ImportError as e:
    import logging
    logging.warning(f"SupertonicTTS import failed: {e}")
    SupertonicTTS = None  # type: ignore

__all__ = ["CustomTTS", "SupertonicTTS"]
