"""
Care Alert Handler for iOS DataChannel alerts.

Handles 4 alert types:
- device_fall: 기기 낙상 (가속도계/자이로)
- person_fall: 사람 낙상 (카메라 기반)
- loud_voice: 큰 소리/비명
- emotion: 감정 분석 결과
"""
import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Callable, Awaitable, Union

from livekit import rtc

from services.api_client import send_care_alert

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    DEVICE_FALL = "device_fall"
    PERSON_FALL = "person_fall"
    LOUD_VOICE = "loud_voice"
    EMOTION = "emotion"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeviceFallType(str, Enum):
    FREEFALL_IMPACT = "freefall_impact"
    IMPACT = "impact"
    ROTATION_IMPACT = "rotation_impact"
    COMBINATION = "combination"


class PersonFallType(str, Enum):
    RAPID_DESCENT = "rapid_descent"
    FACE_DISAPPEARED = "face_disappeared"
    SIZE_CHANGE = "size_change"


class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


@dataclass
class DeviceFallPayload:
    impact_magnitude: float
    fall_type: DeviceFallType
    freefall_duration: Optional[float] = None
    max_rotation_rate: Optional[float] = None


@dataclass
class PersonFallPayload:
    detection_type: PersonFallType
    face_y_delta: Optional[float] = None
    delta_time: Optional[float] = None
    last_face_position: Optional[dict] = None


@dataclass
class LoudVoicePayload:
    level: float
    decibel: float
    duration: float
    possible_cause: Optional[str] = None


@dataclass
class EmotionPayload:
    emotion: EmotionType
    confidence: float
    intensity: Optional[float] = None
    previous_emotion: Optional[str] = None
    analysis_interval: float = 3.0


PayloadType = Union[DeviceFallPayload, PersonFallPayload, LoudVoicePayload, EmotionPayload]


@dataclass
class CareAlert:
    timestamp: int
    alert_type: AlertType
    severity: AlertSeverity
    payload: PayloadType
    raw_payload: dict  # Original payload for API forwarding


class CareAlertHandler:
    """Handles care alerts from iOS DataChannel."""

    TOPIC = "care_alert"

    # Alert response templates
    RESPONSES: dict = {
        AlertType.DEVICE_FALL: {
            AlertSeverity.CRITICAL: "어르신, 혹시 괜찮으세요? 휴대폰이 떨어진 것 같은데, 다치신 곳은 없으신가요?",
            AlertSeverity.HIGH: "어르신, 괜찮으세요? 무슨 일이 있으신 건 아닌가요?",
        },
        AlertType.PERSON_FALL: {
            AlertSeverity.CRITICAL: "어르신! 괜찮으세요?! 넘어지신 건 아닌가요? 대답해 주세요!",
        },
        AlertType.LOUD_VOICE: {
            AlertSeverity.HIGH: "어르신, 무슨 일이세요? 괜찮으신가요?",
            AlertSeverity.MEDIUM: "어르신, 혹시 무슨 일 있으세요?",
        },
        AlertType.EMOTION: {
            "sad": "어르신, 혹시 기분이 안 좋으신가요? 무슨 걱정되는 일이 있으시면 말씀해 주세요.",
            "fearful": "어르신, 혹시 무서우신 거 있으세요? 제가 도와드릴까요?",
            "angry": "어르신, 혹시 화나시는 일이 있으셨어요? 얘기해 주시면 들을게요.",
        },
    }

    def __init__(
        self,
        room: rtc.Room,
        on_alert_response: Callable[[str], Awaitable[None]],
        ward_id: str,
        call_id: str,
    ):
        """
        Initialize care alert handler.

        Args:
            room: LiveKit room instance
            on_alert_response: Async callback to trigger agent speech
            ward_id: Ward UUID for API correlation
            call_id: Call UUID for API correlation
        """
        self.room = room
        self.on_alert_response = on_alert_response
        self.ward_id = ward_id
        self.call_id = call_id
        self.last_emotion: Optional[EmotionType] = None
        self.emotion_change_count = 0

    def register(self) -> None:
        """Register data_received handler on room."""
        self.room.on("data_received")(self._on_data_received)
        logger.info("CareAlertHandler registered")

    def _on_data_received(self, packet: rtc.DataPacket) -> None:
        """Handle incoming data packets.

        Note: data_received event passes single DataPacket argument.
        participant info is available via packet.participant attribute.
        """
        if packet.topic != self.TOPIC:
            return

        participant_id = packet.participant.identity if packet.participant else "Unknown"
        logger.info(f"[CareAlert] Received {packet.topic}, data_len={len(packet.data)}, from={participant_id}")

        try:
            raw_data = packet.data.decode("utf-8")
            data = json.loads(raw_data)
            alert = self._parse_alert(data)
            if alert:
                self._handle_alert(alert)
            else:
                logger.warning(f"Invalid care alert data: {raw_data[:200]}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse care_alert JSON: {e}, data={packet.data[:200]}")
        except Exception as e:
            logger.error(f"Failed to process care_alert: {e}", exc_info=True)

    def _parse_alert(self, data: dict) -> Optional[CareAlert]:
        """Parse JSON data into CareAlert."""
        try:
            alert_type = AlertType(data["alertType"])
            severity = AlertSeverity(data["severity"])
            payload_data = data.get("data", {}).get("payload", {})

            payload = self._parse_payload(alert_type, payload_data)
            if payload is None:
                return None

            return CareAlert(
                timestamp=data["timestamp"],
                alert_type=alert_type,
                severity=severity,
                payload=payload,
                raw_payload=payload_data,  # Keep original for API forwarding
            )
        except KeyError as e:
            logger.warning(f"Missing required field in alert data: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Invalid enum value in alert data: {e}")
            return None

    def _parse_payload(
        self, alert_type: AlertType, data: dict
    ) -> Optional[PayloadType]:
        """Parse payload based on alert type."""
        if alert_type == AlertType.DEVICE_FALL:
            return DeviceFallPayload(
                impact_magnitude=data.get("impactMagnitude", 0),
                fall_type=DeviceFallType(data.get("fallType", "impact")),
                freefall_duration=data.get("freefallDuration"),
                max_rotation_rate=data.get("maxRotationRate"),
            )
        elif alert_type == AlertType.PERSON_FALL:
            return PersonFallPayload(
                detection_type=PersonFallType(
                    data.get("detectionType", "rapid_descent")
                ),
                face_y_delta=data.get("faceYDelta"),
                delta_time=data.get("deltaTime"),
                last_face_position=data.get("lastFacePosition"),
            )
        elif alert_type == AlertType.LOUD_VOICE:
            return LoudVoicePayload(
                level=data.get("level", 0),
                decibel=data.get("decibel", 0),
                duration=data.get("duration", 0),
                possible_cause=data.get("possibleCause"),
            )
        elif alert_type == AlertType.EMOTION:
            return EmotionPayload(
                emotion=EmotionType(data.get("emotion", "neutral")),
                confidence=data.get("confidence", 0),
                intensity=data.get("intensity"),
                previous_emotion=data.get("previousEmotion"),
                analysis_interval=data.get("analysisInterval", 3.0),
            )
        return None

    def _handle_alert(self, alert: CareAlert) -> None:
        """Handle parsed alert and trigger response if needed."""
        logger.info(f"[CareAlert] {alert.alert_type.value} ({alert.severity.value})")

        response = self._get_response(alert)
        if response:
            asyncio.create_task(self.on_alert_response(response))

        # Publish to backend (Redis first, HTTP fallback)
        asyncio.create_task(
            self._publish_alert(alert, agent_response=response)
        )

    async def _publish_alert(
        self,
        alert: CareAlert,
        agent_response: Optional[str] = None,
    ) -> None:
        """Send alert to backend API for storage and push notification."""
        success = await send_care_alert(
            ward_id=self.ward_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            timestamp=alert.timestamp,
            payload=alert.raw_payload,
            call_id=self.call_id,
            room_name=self.room.name,
            agent_response=agent_response,
        )
        if not success:
            logger.warning(
                f"[CareAlert] Failed to send alert to API: "
                f"ward={self.ward_id} type={alert.alert_type.value}"
            )

    def _get_response(self, alert: CareAlert) -> Optional[str]:
        """Get appropriate response for alert."""
        # Fall alerts - always respond
        if alert.alert_type in (AlertType.DEVICE_FALL, AlertType.PERSON_FALL):
            responses = self.RESPONSES.get(alert.alert_type, {})
            return responses.get(alert.severity) or responses.get(AlertSeverity.HIGH)

        # Loud voice - respond for high severity
        if alert.alert_type == AlertType.LOUD_VOICE:
            if alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
                responses = self.RESPONSES.get(alert.alert_type, {})
                return responses.get(alert.severity)

        # Emotion - respond only for significant negative emotions
        if alert.alert_type == AlertType.EMOTION:
            payload: EmotionPayload = alert.payload

            # Only respond if confidence is high enough
            if payload.confidence < 0.7:
                return None

            # Track consecutive same emotion detections
            if payload.emotion != self.last_emotion:
                self.last_emotion = payload.emotion
                self.emotion_change_count = 1  # Reset on emotion change
            else:
                self.emotion_change_count += 1  # Increment on same emotion

            # Only respond to negative emotions after some stability
            if payload.emotion in (
                EmotionType.SAD,
                EmotionType.FEARFUL,
                EmotionType.ANGRY,
            ):
                # Respond after 2 consecutive detections of same emotion
                if self.emotion_change_count >= 2:
                    return self.RESPONSES[AlertType.EMOTION].get(payload.emotion.value)

        return None
