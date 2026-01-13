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
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Callable, Awaitable, Union

from livekit import rtc

from services.api_client import send_care_alert, acknowledge_care_alert

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
    ACKNOWLEDGE_TOPIC = "acknowledge_alert"
    ALERT_RESPONSE_TOPIC = "alert_response"  # Agent → iOS: alert activation notification
    COOLDOWN_SECONDS = 30.0  # Prevent repeated alerts for 30 seconds

    # Alert response templates
    RESPONSES: dict = {
        AlertType.DEVICE_FALL: {
            AlertSeverity.CRITICAL: "어르신, 혹시 괜찮으세요? 휴대폰이 떨어진 것 같은데, 다치신 곳은 없으신가요?",
            AlertSeverity.HIGH: "어르신, 괜찮으세요? 무슨 일이 있으신 건 아닌가요?",
        },
        AlertType.PERSON_FALL: {
            AlertSeverity.CRITICAL: "어르신! 괜찮으세요?! 혹시 무슨 일 있으세요?",
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
        self.last_alert_id: Optional[str] = None  # Store last alert for acknowledge
        self.last_alert_time: float = 0.0  # Timestamp of last handled alert

    def register(self) -> None:
        """Register data_received handler on room."""
        self.room.on("data_received")(self._on_data_received)
        logger.info(f"CareAlertHandler registered (topics: {self.TOPIC}, {self.ACKNOWLEDGE_TOPIC})")

    def _on_data_received(self, packet: rtc.DataPacket) -> None:
        """Handle incoming data packets.

        Note: data_received event passes single DataPacket argument.
        participant info is available via packet.participant attribute.
        """
        participant_id = packet.participant.identity if packet.participant else "Unknown"

        # Handle care_alert topic
        if packet.topic == self.TOPIC:
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
            return

        # Handle acknowledge_alert topic
        if packet.topic == self.ACKNOWLEDGE_TOPIC:
            logger.info(f"[AcknowledgeAlert] Received {packet.topic}, data_len={len(packet.data)}, from={participant_id}")
            try:
                raw_data = packet.data.decode("utf-8")
                data = json.loads(raw_data)
                self._handle_acknowledge(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse acknowledge_alert JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to process acknowledge_alert: {e}", exc_info=True)
            return

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

        # Cooldown check to prevent repeated TTS (debounce)
        now = time.time()
        time_since_last = now - self.last_alert_time
        
        should_respond = True
        if time_since_last < self.COOLDOWN_SECONDS:
            logger.info(f"[CareAlert] Skipping TTS due to cooldown ({time_since_last:.1f}s < {self.COOLDOWN_SECONDS}s)")
            should_respond = False
        else:
            self.last_alert_time = now

        response = self._get_response(alert)
        if response and should_respond:
            asyncio.create_task(self.on_alert_response(response))

        # Notify iOS app to show alert banner
        if response and should_respond:
            asyncio.create_task(self._notify_ios_alert(alert, response))

        # Publish to backend (Redis first, HTTP fallback)
        # We still publish even if cooldown is active, to keep record
        asyncio.create_task(
            self._publish_alert(alert, agent_response=response if should_respond else None)
        )

    async def _notify_ios_alert(self, alert: CareAlert, agent_response: str) -> None:
        """Send alert notification to iOS app via DataChannel.

        iOS app can display a banner/toast when receiving this message.
        """
        try:
            payload = json.dumps({
                "alertType": alert.alert_type.value,
                "severity": alert.severity.value,
                "agentResponse": agent_response,
                "timestamp": alert.timestamp,
            })
            await self.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic=self.ALERT_RESPONSE_TOPIC,
            )
            logger.info(f"[CareAlert] Notified iOS: {alert.alert_type.value} ({alert.severity.value})")
        except Exception as e:
            logger.error(f"[CareAlert] Failed to notify iOS: {e}")

    async def _publish_alert(
        self,
        alert: CareAlert,
        agent_response: Optional[str] = None,
    ) -> None:
        """Send alert to backend API for storage and push notification."""
        alert_id = await send_care_alert(
            ward_id=self.ward_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            timestamp=alert.timestamp,
            payload=alert.raw_payload,
            call_id=self.call_id,
            room_name=self.room.name,
            agent_response=agent_response,
        )
        if alert_id:
            self.last_alert_id = alert_id
            logger.info(f"[CareAlert] Stored last_alert_id={alert_id} for acknowledge")
        else:
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

    def _handle_acknowledge(self, data: dict) -> None:
        """Handle acknowledge_alert signal from iOS.

        iOS sends this when user says "I'm okay" or presses acknowledge button.
        This triggers API call to dismiss the alert and remove Web UI danger indicator.
        """
        timestamp = data.get("timestamp", 0)
        ward_id = data.get("wardId", self.ward_id)

        logger.info(f"[AcknowledgeAlert] Processing: ward={ward_id} timestamp={timestamp}")

        # Reset cooldown so next alert can trigger TTS immediately
        self.last_alert_time = 0.0
        logger.info("[AcknowledgeAlert] Cooldown reset - next alert will trigger TTS")

        # Use last stored alertId or find from API
        alert_id = self.last_alert_id

        if not alert_id:
            logger.warning("[AcknowledgeAlert] No last_alert_id stored, cannot acknowledge")
            return

        # Call API to acknowledge
        asyncio.create_task(self._publish_acknowledge(alert_id))

    async def _publish_acknowledge(self, alert_id: str) -> None:
        """Send acknowledge request to backend API."""
        success = await acknowledge_care_alert(alert_id)
        if success:
            logger.info(f"[AcknowledgeAlert] Successfully acknowledged: alertId={alert_id}")
            self.last_alert_id = None  # Clear after successful acknowledge
        else:
            logger.warning(f"[AcknowledgeAlert] Failed to acknowledge: alertId={alert_id}")
