"""API client for backend communication."""
import os
import logging
from typing import Optional
from urllib.parse import quote

import httpx

from constants import TIMEOUT_CALL_CONTEXT

logger = logging.getLogger(__name__)

# API configuration
API_BASE = os.getenv("API_BASE_URL")
API_INTERNAL_TOKEN = os.getenv("API_INTERNAL_TOKEN")


def get_auth_headers() -> dict:
    """Get authentication headers for internal API calls."""
    headers = {"Content-Type": "application/json"}
    if API_INTERNAL_TOKEN:
        headers["Authorization"] = f"Bearer {API_INTERNAL_TOKEN}"
    return headers


async def fetch_call_context(room_name: str) -> Optional[dict]:
    """Resolve call metadata from the backend using the LiveKit room name."""
    if not room_name:
        return None

    try:
        encoded_room = quote(room_name, safe='')
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE}/v1/calls/room/{encoded_room}/context",
                headers=get_auth_headers(),
                timeout=TIMEOUT_CALL_CONTEXT,
            )
            response.raise_for_status()
            context = response.json()
            logger.info(f"Resolved call context for room={room_name}: {context}")
            return context
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response else 'unknown'
        if status == 404:
            logger.warning(f"No call record found for room={room_name}")
        else:
            logger.error(f"Call context request failed room={room_name} status={status}")
    except Exception as exc:
        logger.error(f"Call context request error room={room_name}: {exc}")

    return None


async def notify_call_end(call_id: str, ward_id: str) -> bool:
    """Notify backend that call has ended (fallback when Redis fails)."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/v1/calls/end",
                json={"callId": call_id, "wardId": ward_id},
                headers=get_auth_headers(),
                timeout=5.0,
            )
            logger.info(f"Call end notified via API: call={call_id}")
            return True
    except Exception as e:
        logger.error(f"Failed to notify call end via API: {e}")
        return False
