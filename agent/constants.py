"""
Shared constants for voice agent services.

Centralized configuration to prevent duplication and inconsistencies.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo

# Redis Pub/Sub channel names
TRANSCRIPT_CHANNEL = "transcripts"
TTS_CHANNEL = "tts_transcripts"
CALL_END_CHANNEL = "call_end"
GREETING_CHANNEL_PREFIX = "greeting:ward:"  # Channel: greeting:ward:{wardId}

logger = logging.getLogger(__name__)

# Timezone configuration
# Use IANA timezone names to support DST and avoid hardcoded offsets.
def _validate_timezone(value: str, fallback: str) -> str:
    try:
        ZoneInfo(value)
        return value
    except Exception:
        logger.warning(
            "Invalid AGENT_TIMEZONE '%s'. Falling back to '%s'.",
            value,
            fallback,
        )
        return fallback


AGENT_TIMEZONE = _validate_timezone(
    os.getenv("AGENT_TIMEZONE", "Asia/Seoul"),
    "Asia/Seoul",
)
AGENT_TZINFO = ZoneInfo(AGENT_TIMEZONE)

# Timeouts (seconds)
TIMEOUT_CALL_CONTEXT = 10.0       # RAG context/search default for PGVector queries
TIMEOUT_RAG_INDEXING = 10.0       # RAG indexing trigger (embedding generation)
TIMEOUT_CALL_END = 10.0           # Call end notification (AI analysis)
TIMEOUT_RAG_CONTEXT_WARMUP = 2.0  # Fast context fetch during call startup
TIMEOUT_RAG_SEARCH_QUICK = 3.0    # Quick RAG search during conversation
TIMEOUT_GREETING_FETCH = 5.0      # Wait for personalized greeting from Redis Pub/Sub (allows time for LLM generation)

# Redis retry configuration
REDIS_MAX_RETRIES = 5
REDIS_RETRY_DELAY = 2.0  # seconds
REDIS_RETRY_BACKOFF = 2.0  # exponential backoff multiplier

def _parse_float_env(
    name: str,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s '%s'; using default %s.", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning("%s below %s; using default %s.", name, min_value, default)
        return default
    if max_value is not None and value > max_value:
        logger.warning("%s above %s; using default %s.", name, max_value, default)
        return default
    return value


def _parse_int_env(
    name: str,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s '%s'; using default %s.", name, raw, default)
        return default
    if min_value is not None and value < min_value:
        logger.warning("%s below %s; using default %s.", name, min_value, default)
        return default
    if max_value is not None and value > max_value:
        logger.warning("%s above %s; using default %s.", name, max_value, default)
        return default
    return value


@dataclass(frozen=True)
class RAGConfig:
    """Validated RAG configuration."""

    memory_similarity_threshold: float
    child_chunk_size: int
    child_chunk_overlap: int
    window_context_chars: int

    @classmethod
    def from_env(cls) -> "RAGConfig":
        memory_similarity_threshold = _parse_float_env(
            "RAG_MEMORY_SIMILARITY_THRESHOLD",
            0.35,
            min_value=0.0,
            max_value=1.0,
        )
        child_chunk_size = _parse_int_env(
            "RAG_CHILD_CHUNK_SIZE",
            200,
            min_value=1,
        )
        child_chunk_overlap = _parse_int_env(
            "RAG_CHILD_CHUNK_OVERLAP",
            50,
            min_value=0,
        )
        if child_chunk_overlap >= child_chunk_size:
            adjusted = max(child_chunk_size - 1, 0)
            logger.warning(
                "RAG_CHILD_CHUNK_OVERLAP (%s) >= RAG_CHILD_CHUNK_SIZE (%s); using %s.",
                child_chunk_overlap,
                child_chunk_size,
                adjusted,
            )
            child_chunk_overlap = adjusted
        window_context_chars = _parse_int_env(
            "RAG_WINDOW_CONTEXT_CHARS",
            150,
            min_value=1,
        )
        return cls(
            memory_similarity_threshold=memory_similarity_threshold,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            window_context_chars=window_context_chars,
        )


RAG_CONFIG = RAGConfig.from_env()

# Backwards-compatible exports
MEMORY_SIMILARITY_THRESHOLD = RAG_CONFIG.memory_similarity_threshold
RAG_CHILD_CHUNK_SIZE = RAG_CONFIG.child_chunk_size
RAG_CHILD_CHUNK_OVERLAP = RAG_CONFIG.child_chunk_overlap
RAG_WINDOW_CONTEXT_CHARS = RAG_CONFIG.window_context_chars
