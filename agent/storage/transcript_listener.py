"""
Transcript Storage Service - Handles storing and post-processing call transcripts.

This service:
- Subscribes to Redis Pub/Sub for transcript events
- Stores transcripts to Redis
- Triggers RAG indexing when calls end
- Triggers call analysis
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

# Add parent directory to path for imports when running as standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import redis.asyncio as redis_async

from constants import (
    TRANSCRIPT_CHANNEL,
    TTS_CHANNEL,
    CALL_END_CHANNEL,
    TIMEOUT_RAG_INDEXING,
    TIMEOUT_CALL_END,
    REDIS_MAX_RETRIES,
    REDIS_RETRY_DELAY,
    REDIS_RETRY_BACKOFF,
)

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL")

# API configuration
API_BASE = os.getenv("API_BASE_URL")
API_INTERNAL_TOKEN = os.getenv("API_INTERNAL_TOKEN")


def _get_auth_headers() -> dict:
    """Get authentication headers for internal API calls."""
    headers = {"Content-Type": "application/json"}
    if API_INTERNAL_TOKEN:
        headers["Authorization"] = f"Bearer {API_INTERNAL_TOKEN}"
    return headers


async def save_transcript_to_redis(redis_client, call_id: str, speaker: str, text: str, timestamp: str):
    """Save transcript entry to Redis."""
    try:
        transcript_entry = {
            "speaker": speaker,
            "text": text,
            "timestamp": timestamp,
        }
        redis_key = f"call:{call_id}:transcripts"
        pipe = redis_client.pipeline()
        pipe.rpush(redis_key, json.dumps(transcript_entry, ensure_ascii=False))
        pipe.expire(redis_key, 3600 * 24)  # 24 hours
        await pipe.execute()
        logger.info(f"Saved to Redis: {call_id} - {speaker} - {text[:50]}")
    except Exception as e:
        logger.error(f"Failed to save transcript to Redis: {e}")


async def trigger_rag_indexing(call_id: str, ward_id: str):
    """Trigger RAG indexing after session ends."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/v1/rag/index",
                json={
                    "callId": call_id,
                    "wardId": ward_id,
                },
                headers=_get_auth_headers(),
                timeout=TIMEOUT_RAG_INDEXING,
            )
            logger.info(f"RAG indexing triggered: call={call_id}")
    except Exception as e:
        logger.error(f"RAG indexing trigger failed: {e}")


async def trigger_call_end(call_id: str):
    """Inform API that the call ended so it can finalize state and summaries."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/v1/calls/end",
                json={"callId": call_id},
                headers=_get_auth_headers(),
                timeout=TIMEOUT_CALL_END,
            )
            logger.info(f"Call end notified: call={call_id}")
    except Exception as e:
        logger.error(f"Call end trigger failed: {e}")


async def handle_transcript_message(redis_client, message: dict):
    """Handle incoming transcript message (STT)."""
    try:
        data = json.loads(message["data"])
        call_id = data.get("call_id")
        speaker = data.get("speaker")
        text = data.get("text")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())

        if not call_id or not speaker or not text:
            logger.warning(f"Invalid transcript message: {data}")
            return

        await save_transcript_to_redis(redis_client, call_id, speaker, text, timestamp)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode transcript message: {e}")
    except Exception as e:
        logger.error(f"Error handling transcript message: {e}")


async def handle_tts_message(redis_client, message: dict):
    """Handle incoming TTS message (AI responses)."""
    try:
        data = json.loads(message["data"])
        call_id = data.get("call_id")
        text = data.get("text")
        timestamp = data.get("timestamp", datetime.utcnow().isoformat())

        if not call_id or not text:
            logger.warning(f"Invalid TTS message: {data}")
            return

        # TTS messages are always from the AI assistant
        await save_transcript_to_redis(redis_client, call_id, "assistant", text, timestamp)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode TTS message: {e}")
    except Exception as e:
        logger.error(f"Error handling TTS message: {e}")


async def handle_call_end_message(redis_client, message: dict):
    """
    Handle call end notification.

    Process flow:
    1. Verify transcripts exist in Redis (with retry)
    2. Trigger call end → AI analysis
    3. Trigger RAG indexing (automatic via AI service)

    Note: Agent waits 5s after session end to ensure transcripts are saved.
    """
    try:
        data = json.loads(message["data"])
        call_id = data.get("call_id")
        ward_id = data.get("ward_id")

        if not call_id or not ward_id:
            logger.warning(f"❌ Invalid call_end message (missing call_id/ward_id): {data}")
            return

        logger.info(f"📞 Call ended: {call_id}, starting post-processing...")

        # Verify transcripts exist in Redis
        transcript_count = 0
        redis_key = f"call:{call_id}:transcripts"

        try:
            # Retry up to 10 times (5 second total)
            for attempt in range(10):
                try:
                    transcript_count = await redis_client.llen(redis_key)
                    if transcript_count and transcript_count > 0:
                        logger.info(f"✅ Found {transcript_count} transcripts for call={call_id}")
                        break
                    logger.debug(f"Waiting for transcripts... ({attempt + 1}/10)")
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Redis check error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(0.5)

            if transcript_count == 0:
                logger.warning(
                    f"⚠️  No transcripts found for call={call_id} after retries. "
                    f"AI analysis and RAG indexing may have empty data."
                )
        except Exception as e:
            logger.error(f"❌ Transcript verification failed: {e}. Proceeding anyway...")

        # Run post-processing tasks
        logger.info(f"🚀 Triggering post-processing for call={call_id}...")

        tasks = [
            trigger_call_end(call_id),
            trigger_rag_indexing(call_id, ward_id),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        for i, result in enumerate(results):
            task_name = ["call_end", "rag_indexing"][i]
            if isinstance(result, Exception):
                logger.error(f"❌ {task_name} failed: {result}")

        logger.info(
            f"✅ Post-processing done for call={call_id} "
            f"(transcripts={transcript_count})"
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode call_end JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in call_end handler: {e}", exc_info=True)


async def connect_to_redis_with_retry():
    """Connect to Redis with exponential backoff retry logic."""
    for attempt in range(REDIS_MAX_RETRIES):
        try:
            redis_client = redis_async.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            logger.info(f"Successfully connected to Redis (attempt {attempt + 1}/{REDIS_MAX_RETRIES})")
            return redis_client
        except Exception as e:
            retry_delay = REDIS_RETRY_DELAY * (REDIS_RETRY_BACKOFF ** attempt)
            logger.warning(f"Redis connection attempt {attempt + 1}/{REDIS_MAX_RETRIES} failed: {e}")
            if attempt < REDIS_MAX_RETRIES - 1:
                logger.info(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Failed to connect to Redis after all retries")
                raise


async def main():
    """Main service loop with automatic reconnection."""
    logger.info("Starting Transcript Storage Service...")
    logger.info(f"Connecting to Redis: {REDIS_URL}")

    while True:
        redis_client = None
        pubsub = None

        try:
            # Connect to Redis with retry logic
            redis_client = await connect_to_redis_with_retry()

            # Subscribe to channels
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(TRANSCRIPT_CHANNEL, TTS_CHANNEL, CALL_END_CHANNEL)
            logger.info(f"Subscribed to channels: {TRANSCRIPT_CHANNEL}, {TTS_CHANNEL}, {CALL_END_CHANNEL}")

            # Listen for messages
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                channel = message["channel"]

                if channel == TRANSCRIPT_CHANNEL:
                    await handle_transcript_message(redis_client, message)
                elif channel == TTS_CHANNEL:
                    await handle_tts_message(redis_client, message)
                elif channel == CALL_END_CHANNEL:
                    await handle_call_end_message(redis_client, message)

        except KeyboardInterrupt:
            logger.info("Service stopped by user")
            break
        except Exception as e:
            logger.error(f"Connection error: {e}")
            logger.info(f"Reconnecting in {REDIS_RETRY_DELAY}s...")
            await asyncio.sleep(REDIS_RETRY_DELAY)
        finally:
            # Cleanup
            if pubsub:
                try:
                    await pubsub.unsubscribe(TRANSCRIPT_CHANNEL, TTS_CHANNEL, CALL_END_CHANNEL)
                    await pubsub.close()
                except Exception as e:
                    logger.error(f"Error closing pubsub: {e}")
            if redis_client:
                try:
                    await redis_client.close()
                except Exception as e:
                    logger.error(f"Error closing redis client: {e}")

    logger.info("Service shutdown complete")


if __name__ == "__main__":
    print("=" * 50)
    print("TRANSCRIPT STORAGE SERVICE")
    print("=" * 50)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
