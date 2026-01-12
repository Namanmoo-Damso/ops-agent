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
    GREETING_CHANNEL_PREFIX,
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


async def subscribe_to_greeting(ward_id: str, callback, timeout: float = 5.0) -> None:
    """
    Subscribe to greeting channel and invoke callback when greeting arrives.
    
    This implements a Push-based approach where the agent subscribes to a Redis
    channel and receives the personalized greeting when the backend publishes it.
    
    Channel: greeting:ward:{wardId}
    
    Args:
        ward_id: Ward UUID
        callback: Async function to call with greeting text when received
        timeout: Maximum time to wait for greeting (default: 5 seconds)
    
    Flow:
        1. Subscribe to greeting:ward:{wardId}
        2. Wait for message (with timeout)
        3. When message arrives, invoke callback with greeting text
        4. Auto-unsubscribe after first message or timeout
    """
    client = await get_redis_client()
    if not client:
        logger.warning(f"Redis client unavailable for greeting subscription (ward={ward_id})")
        return

    channel_name = f"{GREETING_CHANNEL_PREFIX}{ward_id}"
    pubsub = client.pubsub()
    
    try:
        # Subscribe to the greeting channel
        await pubsub.subscribe(channel_name)
        logger.info(f"📡 Subscribed to greeting channel: {channel_name}")
        
        # Wait for greeting message with timeout
        try:
            async with asyncio.timeout(timeout):
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        greeting_text = message["data"]
                        logger.info(f"✅ Received greeting from channel (ward={ward_id}, length={len(greeting_text)})")
                        
                        # Invoke callback with greeting
                        await callback(greeting_text)
                        
                        # Unsubscribe after receiving first message
                        break
        except asyncio.TimeoutError:
            logger.info(f"⏱️  Greeting subscription timed out after {timeout}s (ward={ward_id})")
    
    except Exception as e:
        logger.error(f"❌ Error in greeting subscription (ward={ward_id}): {e}")
    
    finally:
        # Clean up subscription
        try:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            logger.debug(f"Unsubscribed from greeting channel: {channel_name}")
        except Exception as e:
            logger.error(f"Error closing pubsub connection: {e}")

