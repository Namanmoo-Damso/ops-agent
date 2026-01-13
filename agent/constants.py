"""
Shared constants for voice agent services.

Centralized configuration to prevent duplication and inconsistencies.
"""
import os

# Redis Pub/Sub channel names
TRANSCRIPT_CHANNEL = "transcripts"
TTS_CHANNEL = "tts_transcripts"
CALL_END_CHANNEL = "call_end"
GREETING_CHANNEL_PREFIX = "greeting:ward:"  # Channel: greeting:ward:{wardId}

# Timezone configuration
# Use IANA timezone names to support DST and avoid hardcoded offsets.
AGENT_TIMEZONE = os.getenv("AGENT_TIMEZONE", "Asia/Seoul")

# Timeouts (seconds)
TIMEOUT_CALL_CONTEXT = 10.0       # RAG context/search default for PGVector queries
TIMEOUT_RAG_INDEXING = 10.0       # RAG indexing trigger (embedding generation)
TIMEOUT_CALL_END = 10.0           # Call end notification (AI analysis)
TIMEOUT_RAG_CONTEXT_WARMUP = 2.0  # Fast context fetch during call startup
TIMEOUT_RAG_SEARCH_QUICK = 3.0    # Quick RAG search during conversation
TIMEOUT_GREETING_FETCH = 5.0      # Wait for personalized greeting from Redis Pub/Sub (allows time for LLM generation)

# Similarity threshold for memory recall (lowered from 0.5 to 0.4 to improve recall coverage).
# Tune via env var if relevance drops: RAG_MEMORY_SIMILARITY_THRESHOLD.
MEMORY_SIMILARITY_THRESHOLD = float(
    os.getenv("RAG_MEMORY_SIMILARITY_THRESHOLD", "0.4")
)

# Redis retry configuration
REDIS_MAX_RETRIES = 5
REDIS_RETRY_DELAY = 2.0  # seconds
REDIS_RETRY_BACKOFF = 2.0  # exponential backoff multiplier
