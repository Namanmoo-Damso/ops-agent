"""Redis Pub/Sub service for transcript and event publishing."""
import asyncio
import logging
import json
from datetime import datetime
from typing import Optional

import redis
import redis.asyncio as redis_async

from config import validate_env_vars
from constants import (
    TRANSCRIPT_CHANNEL,
    CALL_END_CHANNEL,
    REDIS_MAX_RETRIES,
    REDIS_RETRY_DELAY,
    REDIS_RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)

# Redis client singleton
_redis_client: Optional[redis_async.Redis] = None
_redis_init_lock = asyncio.Lock()


async def get_redis_client() -> Optional[redis_async.Redis]:
    """Get or initialize the Redis client with retry logic."""
    global _redis_client

    if _redis_client:
        return _redis_client

    async with _redis_init_lock:
        if _redis_client:
            return _redis_client

        env_config = validate_env_vars()

        for attempt in range(REDIS_MAX_RETRIES):
            try:
                _redis_client = redis_async.from_url(
                    env_config["REDIS_URL"],
                    decode_responses=True,
                )
                await _redis_client.ping()
                logger.info("Successfully connected to Redis for Pub/Sub")
                return _redis_client
            except redis.exceptions.ConnectionError as e:
                retry_delay = REDIS_RETRY_DELAY * (REDIS_RETRY_BACKOFF ** attempt)
                logger.warning(f"Redis connection attempt {attempt + 1}/{REDIS_MAX_RETRIES} failed: {e}")
                if attempt < REDIS_MAX_RETRIES - 1:
                    logger.info(f"Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Failed to connect to Redis after all retries")
                    _redis_client = None
            except Exception as e:
                logger.error(f"Unexpected error connecting to Redis: {e}")
                _redis_client = None
                break

    return None


async def publish_transcript(call_id: str, speaker: str, text: str) -> bool:
    """Publish transcript event to Redis Pub/Sub."""
    client = await get_redis_client()
    if not client:
        return False

    event = {
        "call_id": call_id,
        "speaker": speaker,
        "text": text,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        await client.publish(TRANSCRIPT_CHANNEL, json.dumps(event, ensure_ascii=False))
        logger.debug(f"Published transcript to Redis: {speaker} - {text}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish transcript to Redis: {e}")
        return False


async def store_transcript_direct(call_id: str, speaker: str, text: str) -> bool:
    """Store transcript directly to Redis (fallback when Pub/Sub fails)."""
    client = await get_redis_client()
    if not client:
        return False

    try:
        transcript_entry = {
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
        }
        redis_key = f"call:{call_id}:transcripts"
        pipe = client.pipeline()
        pipe.rpush(redis_key, json.dumps(transcript_entry, ensure_ascii=False))
        pipe.expire(redis_key, 3600 * 24)  # 24 hours
        await pipe.execute()
        logger.debug(f"Stored transcript directly to Redis: {speaker} - {text}")
        return True
    except Exception as e:
        logger.error(f"Failed to store transcript directly: {e}")
        return False


async def publish_call_end(call_id: str, ward_id: str) -> bool:
    """Publish call end event to Redis Pub/Sub."""
    client = await get_redis_client()
    if not client:
        return False

    event = {
        "call_id": call_id,
        "ward_id": ward_id,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        await client.publish(CALL_END_CHANNEL, json.dumps(event, ensure_ascii=False))
        logger.info(f"Published call_end event to Redis: call={call_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish call_end to Redis: {e}")
        return False
