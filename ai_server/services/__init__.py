"""
AI Server Services - STT, TTS, Turn Detection, and Embedding.
"""

from .embedding_service import (
    EmbeddingService,
    embedding_service,
    get_embedding_service,
)
from .stt_service import STTService, WhisperSingleton
from .tts_service import TTSService
from .turn_detector_service import (
    TurnDetectorService,
    TurnDetectionResult,
    get_turn_detector_service,
    turn_detector_service,
)

__all__ = [
    # Embedding Service
    "EmbeddingService",
    "embedding_service",
    "get_embedding_service",
    # STT/TTS Services
    "STTService",
    "WhisperSingleton",
    "TTSService",
    # Turn Detector Service
    "TurnDetectorService",
    "TurnDetectionResult",
    "get_turn_detector_service",
    "turn_detector_service",
]
