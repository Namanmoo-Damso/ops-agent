"""API client for backend communication."""
import os
import logging
from typing import Optional
from urllib.parse import quote

import httpx

from ..constants import TIMEOUT_CALL_CONTEXT

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


async def send_care_alert(
    ward_id: str,
    alert_type: str,
    severity: str,
    timestamp: int,
    payload: dict,
    call_id: Optional[str] = None,
    room_name: Optional[str] = None,
    agent_response: Optional[str] = None,
) -> Optional[str]:
    """
    Send care alert to backend API.

    Args:
        ward_id: Ward UUID
        alert_type: Alert type (device_fall, person_fall, loud_voice, emotion)
        severity: Severity level (low, medium, high, critical)
        timestamp: Event timestamp from iOS (Unix ms)
        payload: Raw payload data from iOS
        call_id: Optional call UUID for correlation
        room_name: Optional LiveKit room name
        agent_response: Optional agent response message

    Returns:
        alertId if sent successfully, None otherwise
    """
    if not API_BASE:
        logger.error("API_BASE_URL not configured")
        return None

    # Build request body matching CreateCareAlertDto
    request_body = {
        "timestamp": timestamp,
        "alertType": alert_type,
        "severity": severity,
        "data": {
            "type": alert_type,
            "payload": payload,
        },
        "source": "agent",
    }

    # Add optional agent correlation fields
    if call_id:
        request_body["callId"] = call_id
    if room_name:
        request_body["roomName"] = room_name
    if agent_response:
        request_body["agentResponse"] = agent_response

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE}/v1/care-alerts",
                json=request_body,
                headers={
                    **get_auth_headers(),
                    "X-Ward-Id": ward_id,  # Ward ID in header for auth
                },
                timeout=5.0,
            )
            response.raise_for_status()
            result = response.json()
            alert_id = result.get("alertId")
            logger.info(f"Care alert sent via API: ward={ward_id} type={alert_type} alertId={alert_id}")
            return alert_id
    except httpx.HTTPStatusError as e:
        logger.error(f"Care alert API error: status={e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Failed to send care alert via API: {e}")
        return None


async def acknowledge_care_alert(alert_id: str) -> bool:
    """
    Acknowledge (dismiss) a care alert via backend API.

    Args:
        alert_id: Alert UUID to acknowledge

    Returns:
        True if acknowledged successfully, False otherwise
    """
    if not API_BASE:
        logger.error("API_BASE_URL not configured")
        return False

    if not alert_id:
        logger.warning("acknowledge_care_alert called with empty alert_id")
        return False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{API_BASE}/v1/guardians/alerts/{alert_id}/acknowledge",
                headers=get_auth_headers(),
                timeout=5.0,
            )
            response.raise_for_status()
            logger.info(f"Care alert acknowledged via API: alertId={alert_id}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Acknowledge alert API error: alertId={alert_id} status={e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Failed to acknowledge care alert via API: {e}")
        return False
