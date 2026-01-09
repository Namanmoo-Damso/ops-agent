"""
Shared constants for voice agent services.

Centralized configuration to prevent duplication and inconsistencies.
"""

# Redis Pub/Sub channel names
TRANSCRIPT_CHANNEL = "transcripts"
TTS_CHANNEL = "tts_transcripts"
CALL_END_CHANNEL = "call_end"

# Timeouts (seconds)
TIMEOUT_CALL_CONTEXT = 5.0
TIMEOUT_RAG_INDEXING = 5.0
TIMEOUT_CALL_END = 5.0

# Redis retry configuration
REDIS_MAX_RETRIES = 5
REDIS_RETRY_DELAY = 2.0  # seconds
REDIS_RETRY_BACKOFF = 2.0  # exponential backoff multiplier
