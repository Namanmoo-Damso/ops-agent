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
    risk_level: Optional[str] = None,
    risk_score: Optional[float] = None,
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
        risk_level: Optional risk level (normal, caution, critical)
        risk_score: Optional risk score (0.0 to 1.0)

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
    if risk_level:
        request_body["riskLevel"] = risk_level
    if risk_score is not None:
        request_body["riskScore"] = risk_score

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


async def clear_room_danger(room_name: str) -> bool:
    """
    Clear danger state for a room when agent session starts.

    Args:
        room_name: LiveKit room name

    Returns:
        True if cleared successfully, False otherwise
    """
    if not API_BASE:
        logger.error("API_BASE_URL not configured")
        return False

    if not room_name:
        logger.warning("clear_room_danger called with empty room_name")
        return False

    encoded_room = quote(room_name, safe='')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE}/v1/livekit/rooms/{encoded_room}/danger",
                json={"isDanger": False},
                headers=get_auth_headers(),
                timeout=3.0,
            )
            response.raise_for_status()
            logger.info(f"Room danger state cleared: room={room_name}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Clear room danger API error: room={room_name} status={e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Failed to clear room danger: room={room_name} error={e}")
        return False


async def escalate_care_alert(alert_id: str) -> bool:
    """
    Escalate a care alert from caution to critical via backend API.

    Called when iOS user presses "도움이 필요해요" (need help) button.
    The server will update the alert status and set escalatedFromCaution flag
    in room metadata to prevent duplicate alerts.

    Args:
        alert_id: Alert UUID to escalate

    Returns:
        True if escalated successfully, False otherwise
    """
    if not API_BASE:
        logger.error("API_BASE_URL not configured")
        return False

    if not alert_id:
        logger.warning("escalate_care_alert called with empty alert_id")
        return False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{API_BASE}/v1/guardians/alerts/{alert_id}/escalate",
                headers=get_auth_headers(),
                timeout=5.0,
            )
            response.raise_for_status()
            logger.info(f"Care alert escalated via API: alertId={alert_id}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Escalate alert API error: alertId={alert_id} status={e.response.status_code}"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to escalate care alert via API: {e}")
        return False


async def send_sensor_emotion(
    ward_id: str,
    timestamp: int,
    emotion: str,
    confidence: float,
    intensity: Optional[float] = None,
) -> bool:
    """
    Send sensor emotion data to backend API for buffering.

    Uses the care-alerts endpoint with alertType='emotion'.
    API buffers emotion data and aggregates every 10 minutes for reports.

    Args:
        ward_id: Ward UUID
        timestamp: Event timestamp (Unix ms)
        emotion: Emotion type (neutral, happy, sad, angry, fearful, disgusted, surprised)
        confidence: Confidence score (0.0 to 1.0)
        intensity: Optional intensity score (0.0 to 1.0)

    Returns:
        True if sent successfully, False otherwise
    """
    if not API_BASE:
        logger.error("API_BASE_URL not configured")
        return False

    # Build request body matching CreateCareAlertDto for emotion buffering
    request_body = {
        "timestamp": timestamp,
        "alertType": "emotion",
        "severity": "low",  # Sensor stream emotions are non-alert (low severity)
        "data": {
            "type": "emotion",
            "payload": {
                "emotion": emotion,
                "confidence": confidence,
            },
        },
        "source": "sensor_stream",  # Mark as sensor stream (not alert)
    }

    # Add optional intensity
    if intensity is not None:
        request_body["data"]["payload"]["intensity"] = intensity

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE}/v1/care-alerts",
                json=request_body,
                headers={
                    **get_auth_headers(),
                    "X-Ward-Id": ward_id,
                },
                timeout=3.0,  # Shorter timeout for high-frequency sensor data
            )
            response.raise_for_status()
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Sensor emotion API error: status={e.response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Failed to send sensor emotion via API: {e}")
        return False
