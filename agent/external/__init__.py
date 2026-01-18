"""External AI Server clients for STT, TTS, and Turn Detection."""

from .external_stt import ExternalSTT
from .external_tts import ExternalTTS
from .external_turn_detector import ExternalTurnDetector

__all__ = ["ExternalSTT", "ExternalTTS", "ExternalTurnDetector"]
