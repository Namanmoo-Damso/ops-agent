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

from ..services.api_client import (
    send_care_alert,
    acknowledge_care_alert,
    escalate_care_alert,
    send_sensor_emotion,
)
from .sensor_detector import SensorDetector, DetectionResult, RiskLevel as DetectorRiskLevel, AlertType as DetectorAlertType

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


class RiskLevel(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
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


# Sensor Stream Payloads (for continuous sensor data from iOS)
@dataclass
class SensorEmotionData:
    """Emotion data from sensor_stream topic."""
    emotion: str  # neutral, happy, sad, angry, fearful, disgusted, surprised
    confidence: float
    intensity: Optional[float] = None


@dataclass
class SensorAudioData:
    """Audio data from sensor_stream topic."""
    level: float
    decibel: float


@dataclass
class SensorMotionData:
    """Motion data from sensor_stream topic."""
    fall_risk: float  # 0.0 ~ 1.0 (iOS 계산)
    acceleration_magnitude: float  # 가속도 크기 (g)
    rotation_magnitude: float  # 회전 크기 (rad/s)
    is_freefalling: bool  # 자유낙하 여부


@dataclass
class SensorFaceData:
    """Face data from sensor_stream topic."""
    is_detected: bool  # 얼굴 감지 여부
    face_y: float  # 현재 얼굴 Y 위치
    y_delta: float  # 최근 0.5초간 Y 변화량 (양수=하강)
    delta_time: float  # Y 변화 측정 시간 (초)
    disappeared_duration: float  # 얼굴 미감지 지속 시간 (초)


@dataclass
class SensorStreamPayload:
    """Combined sensor stream payload from iOS."""
    timestamp: int
    emotion: Optional[SensorEmotionData] = None
    audio: Optional[SensorAudioData] = None
    motion: Optional[SensorMotionData] = None
    face: Optional[SensorFaceData] = None


@dataclass
class CareAlert:
    timestamp: int
    alert_type: AlertType
    severity: AlertSeverity
    payload: PayloadType
    raw_payload: dict  # Original payload for API forwarding
    risk_level: RiskLevel = RiskLevel.CRITICAL  # Default for backward compatibility
    risk_score: float = 1.0  # Default for backward compatibility


class CareAlertHandler:
    """Handles care alerts from iOS DataChannel."""

    TOPIC = "care_alert"
    ACKNOWLEDGE_TOPIC = "acknowledge_alert"
    ALERT_RESPONSE_TOPIC = "alert_response"  # Agent → iOS: alert activation notification
    SENSOR_STREAM_TOPIC = "sensor_stream"  # iOS → Agent: continuous sensor data
    SENSOR_STATUS_TOPIC = "sensor_status"  # Agent → Web: real-time sensor status
    HELP_REQUEST_TOPIC = "help_request"  # iOS → Agent: user help request (SOS button)
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

        # Sensor detector for threshold-based detection
        self.sensor_detector = SensorDetector()

    def register(self) -> None:
        """Register data_received handler on room."""
        self.room.on("data_received")(self._on_data_received)
        logger.info(
            f"CareAlertHandler registered (topics: {self.TOPIC}, {self.ACKNOWLEDGE_TOPIC}, "
            f"{self.SENSOR_STREAM_TOPIC}, {self.SENSOR_STATUS_TOPIC}, {self.HELP_REQUEST_TOPIC})"
        )

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
                # Debug: log raw payload for fall detection analysis
                logger.info(f"[CareAlert] Raw payload: {raw_data}")
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

        # Handle sensor_stream topic (continuous sensor data from iOS)
        if packet.topic == self.SENSOR_STREAM_TOPIC:
            try:
                raw_data = packet.data.decode("utf-8")
                data = json.loads(raw_data)
                self._handle_sensor_stream(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse sensor_stream JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to process sensor_stream: {e}", exc_info=True)
            return

        # Handle help_request topic (SOS button from iOS)
        if packet.topic == self.HELP_REQUEST_TOPIC:
            logger.info(f"[HelpRequest] Received {packet.topic}, data_len={len(packet.data)}, from={participant_id}")
            try:
                raw_data = packet.data.decode("utf-8")
                data = json.loads(raw_data)
                self._handle_help_request(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse help_request JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to process help_request: {e}", exc_info=True)
            return

    def _parse_alert(self, data: dict) -> Optional[CareAlert]:
        """Parse JSON data into CareAlert."""
        try:
            alert_type = AlertType(data["alertType"])

            # care_alert 토픽으로 온 emotion은 무시 (sensor_stream에서 처리)
            if alert_type == AlertType.EMOTION:
                logger.info("[CareAlert] Ignoring emotion from care_alert topic (handled via sensor_stream)")
                return None

            severity = AlertSeverity(data["severity"])
            payload_data = data.get("data", {}).get("payload", {})

            payload = self._parse_payload(alert_type, payload_data)
            if payload is None:
                return None

            # Extract riskLevel and riskScore with backward compatibility
            risk_level_str = data.get("riskLevel", "critical")
            try:
                risk_level = RiskLevel(risk_level_str)
            except ValueError:
                logger.warning(f"Invalid riskLevel '{risk_level_str}', defaulting to critical")
                risk_level = RiskLevel.CRITICAL

            risk_score = data.get("riskScore", 1.0)
            if not isinstance(risk_score, (int, float)):
                risk_score = 1.0

            return CareAlert(
                timestamp=data["timestamp"],
                alert_type=alert_type,
                severity=severity,
                payload=payload,
                raw_payload=payload_data,  # Keep original for API forwarding
                risk_level=risk_level,
                risk_score=float(risk_score),
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
        """Handle parsed alert and trigger response if needed.

        riskLevel-based processing:
        - critical: Immediate voice question + iOS Alert + Server storage (existing behavior)
        - caution: Voice status check only (no iOS Alert)
        - normal: No action (iOS should not send this, but handle gracefully)
        """
        logger.info(
            f"[CareAlert] {alert.alert_type.value} ({alert.severity.value}) "
            f"riskLevel={alert.risk_level.value} riskScore={alert.risk_score}"
        )

        # Handle based on riskLevel
        if alert.risk_level == RiskLevel.NORMAL:
            # Normal risk level: no action needed (iOS should not send this)
            logger.info(f"[CareAlert] Skipping normal riskLevel alert")
            return

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

        # Both critical and caution: trigger voice response (TTS)
        if response and should_respond:
            asyncio.create_task(self.on_alert_response(response))

        # Publish to backend and notify iOS (async)
        asyncio.create_task(
            self._publish_and_notify_ios(alert, response, should_respond)
        )

    async def _publish_and_notify_ios(
        self,
        alert: CareAlert,
        response: Optional[str],
        should_respond: bool,
    ) -> None:
        """Publish alert to backend and notify iOS with alertId.

        Must be called as asyncio.create_task() from sync context.
        """
        # Publish to backend FIRST to get alertId
        alert_id = await self._publish_alert(
            alert, agent_response=response if should_respond else None
        )

        # Both caution and critical: notify iOS app to show alert banner (with alertId)
        if alert.risk_level in (RiskLevel.CAUTION, RiskLevel.CRITICAL):
            # Check if escalated from caution (skip duplicate alert)
            if self._is_escalated_from_caution():
                logger.info(
                    f"[CareAlert] Skipping iOS alert - escalated from caution via iOS button"
                )
            else:
                # Always notify iOS for caution/critical, even without TTS response
                await self._notify_ios_alert(alert, response or "", alert_id)
        else:
            # normal: skip iOS alert notification
            logger.info(f"[CareAlert] Skipping iOS alert for normal riskLevel")

    def _is_escalated_from_caution(self) -> bool:
        """Check if alert was escalated from caution via iOS help button.

        Reads room metadata to check for escalatedFromCaution flag.
        This flag is set by the server when iOS user presses "도움이 필요해요" button.

        Returns:
            True if escalated from caution, False otherwise
        """
        try:
            metadata = self.room.metadata
            if not metadata:
                return False

            import json
            metadata_dict = json.loads(metadata)
            escalated = metadata_dict.get("escalatedFromCaution", False)

            if escalated:
                logger.info(
                    f"[CareAlert] Room metadata has escalatedFromCaution=true"
                )
            return escalated
        except json.JSONDecodeError as e:
            logger.warning(f"[CareAlert] Failed to parse room metadata: {e}")
            return False
        except Exception as e:
            logger.warning(f"[CareAlert] Error checking escalation status: {e}")
            return False

    async def _notify_ios_alert(
        self,
        alert: CareAlert,
        agent_response: str,
        alert_id: Optional[str] = None,
    ) -> None:
        """Send alert notification to iOS app via DataChannel.

        iOS app can display a banner/toast when receiving this message.
        iOS should use alertId when responding with acknowledge_alert topic.
        """
        try:
            # Build detection criteria info for UI display
            detection_info = self._build_detection_info(alert)

            payload = json.dumps({
                "alertId": alert_id,  # iOS uses this to respond
                "wardId": self.ward_id,  # ward identifier
                "alertType": alert.alert_type.value,
                "severity": alert.severity.value,
                "riskLevel": alert.risk_level.value,  # caution or critical
                "riskScore": alert.risk_score,
                "agentResponse": agent_response,
                "timestamp": alert.timestamp,
                "detectionInfo": detection_info,
            })
            await self.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic=self.ALERT_RESPONSE_TOPIC,
            )
            logger.info(f"[CareAlert] Notified iOS: alertId={alert_id} type={alert.alert_type.value} ({alert.severity.value})")
        except Exception as e:
            logger.error(f"[CareAlert] Failed to notify iOS: {e}")

    def _build_detection_info(self, alert: CareAlert) -> dict:
        """Build detection criteria information for display.

        Returns info about what triggered the alert and confidence/severity details.
        """
        info = {
            "type": alert.alert_type.value,
            "severity": alert.severity.value,
            "criteria": [],
        }

        try:
            self._populate_detection_criteria(alert, info)
        except Exception as e:
            logger.warning(f"[CareAlert] Failed to build detection criteria: {e}")

        return info

    def _populate_detection_criteria(self, alert: CareAlert, info: dict) -> None:
        """Populate detection criteria based on alert type."""
        if alert.alert_type == AlertType.DEVICE_FALL:
            payload: DeviceFallPayload = alert.payload
            info["criteria"].append({
                "name": "충격 강도",
                "value": f"{payload.impact_magnitude:.1f}",
                "level": "high" if payload.impact_magnitude > 3.0 else "medium" if payload.impact_magnitude > 2.0 else "low",
            })
            info["criteria"].append({
                "name": "낙하 유형",
                "value": payload.fall_type.value,
                "level": "high" if payload.fall_type == DeviceFallType.FREEFALL_IMPACT else "medium",
            })
            if payload.freefall_duration:
                info["criteria"].append({
                    "name": "자유낙하 시간",
                    "value": f"{payload.freefall_duration:.2f}초",
                    "level": "high" if payload.freefall_duration > 0.3 else "medium",
                })

        elif alert.alert_type == AlertType.PERSON_FALL:
            payload: PersonFallPayload = alert.payload
            info["criteria"].append({
                "name": "감지 유형",
                "value": payload.detection_type.value,
                "level": "high",
            })
            if payload.face_y_delta:
                info["criteria"].append({
                    "name": "얼굴 Y축 변화",
                    "value": f"{payload.face_y_delta:.1f}",
                    "level": "high" if abs(payload.face_y_delta) > 100 else "medium",
                })

        elif alert.alert_type == AlertType.LOUD_VOICE:
            payload: LoudVoicePayload = alert.payload
            info["criteria"].append({
                "name": "음량 레벨",
                "value": f"{payload.level:.1f}",
                "level": "high" if payload.level > 0.8 else "medium" if payload.level > 0.6 else "low",
            })
            info["criteria"].append({
                "name": "데시벨",
                "value": f"{payload.decibel:.0f}dB",
                "level": "high" if payload.decibel > 80 else "medium" if payload.decibel > 70 else "low",
            })
            info["criteria"].append({
                "name": "지속 시간",
                "value": f"{payload.duration:.1f}초",
                "level": "high" if payload.duration > 2.0 else "medium",
            })

        elif alert.alert_type == AlertType.EMOTION:
            payload: EmotionPayload = alert.payload
            info["criteria"].append({
                "name": "감정",
                "value": payload.emotion.value,
                "level": "high" if payload.emotion in (EmotionType.FEARFUL, EmotionType.SAD) else "medium",
            })
            info["criteria"].append({
                "name": "신뢰도",
                "value": f"{payload.confidence * 100:.0f}%",
                "level": "high" if payload.confidence > 0.85 else "medium" if payload.confidence > 0.7 else "low",
            })
            if payload.intensity:
                info["criteria"].append({
                    "name": "강도",
                    "value": f"{payload.intensity * 100:.0f}%",
                    "level": "high" if payload.intensity > 0.7 else "medium",
                })

    async def _publish_alert(
        self,
        alert: CareAlert,
        agent_response: Optional[str] = None,
    ) -> Optional[str]:
        """Send alert to backend API for storage and push notification.

        Returns:
            alertId from server if successful, None otherwise
        """
        alert_id = await send_care_alert(
            ward_id=self.ward_id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            timestamp=alert.timestamp,
            payload=alert.raw_payload,
            call_id=self.call_id,
            room_name=self.room.name,
            agent_response=agent_response,
            risk_level=alert.risk_level.value,
            risk_score=alert.risk_score,
        )
        if alert_id:
            self.last_alert_id = alert_id
            logger.info(f"[CareAlert] Stored last_alert_id={alert_id} for acknowledge")
            return alert_id
        else:
            logger.warning(
                f"[CareAlert] Failed to send alert to API: "
                f"ward={self.ward_id} type={alert.alert_type.value}"
            )
            return None

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

            # Only respond if confidence is high enough (critical level)
            if payload.confidence < 0.8:
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

        iOS sends this when user responds to care alert:
        - "괜찮아요" (response: "ok") -> Acknowledge and dismiss alert
        - "도움이 필요해요" (response: "need_help") -> Escalate to critical

        Payload example:
        {
            "alertId": "string",
            "response": "ok" | "need_help",
            "timestamp": 1705123456789,
            "wardId": "ward-uuid"
        }
        """
        timestamp = data.get("timestamp", 0)
        ward_id = data.get("wardId", self.ward_id)
        alert_id = data.get("alertId") or self.last_alert_id
        response = data.get("response", "ok")  # Default to "ok" for backward compatibility

        logger.info(
            f"[AcknowledgeAlert] Processing: ward={ward_id} alertId={alert_id} "
            f"response={response} timestamp={timestamp}"
        )

        if not alert_id:
            logger.warning("[AcknowledgeAlert] No alertId provided and no last_alert_id stored")
            return

        # Reset cooldown so next alert can trigger TTS immediately
        self.last_alert_time = 0.0
        logger.info("[AcknowledgeAlert] Cooldown reset - next alert will trigger TTS")

        # Handle based on response type
        if response == "need_help":
            # User pressed "도움이 필요해요" - escalate to critical
            logger.info(f"[AcknowledgeAlert] User needs help - escalating alert: {alert_id}")
            asyncio.create_task(self._publish_escalate(alert_id))
        else:
            # User pressed "괜찮아요" - acknowledge (dismiss) the alert
            asyncio.create_task(self._publish_acknowledge(alert_id))

    async def _publish_acknowledge(self, alert_id: str) -> None:
        """Send acknowledge request to backend API."""
        success = await acknowledge_care_alert(alert_id)
        if success:
            logger.info(f"[AcknowledgeAlert] Successfully acknowledged: alertId={alert_id}")
            self.last_alert_id = None  # Clear after successful acknowledge
        else:
            logger.warning(f"[AcknowledgeAlert] Failed to acknowledge: alertId={alert_id}")

    async def _publish_escalate(self, alert_id: str) -> None:
        """Send escalate request to backend API.

        Called when user presses "도움이 필요해요" (need help) button.
        Server will update alert status and set escalatedFromCaution flag in room metadata.
        """
        success = await escalate_care_alert(alert_id)
        if success:
            logger.info(f"[AcknowledgeAlert] Successfully escalated: alertId={alert_id}")
            # Do NOT clear last_alert_id - alert is still active (escalated, not dismissed)
        else:
            logger.warning(f"[AcknowledgeAlert] Failed to escalate: alertId={alert_id}")

    def _handle_sensor_stream(self, data: dict) -> None:
        """Handle sensor_stream data from iOS.

        iOS sends continuous sensor data for:
        - emotion: Face emotion analysis (neutral, happy, sad, etc.)
        - audio: Audio level monitoring
        - motion: Accelerometer/gyroscope data
        - face: Face detection status

        Each sensor is analyzed using threshold-based detection.
        When thresholds are exceeded, care_alert is generated.
        Also publishes sensor_status to Web for real-time monitoring.
        """
        # DEBUG: Log full raw data to see what iOS sends
        logger.info(f"[SensorStream] RAW data: {data}")

        timestamp = data.get("timestamp", 0) or int(time.time() * 1000)

        # Build sensor_status payload for Web
        sensor_status: dict = {
            "timestamp": timestamp,
            "wardId": self.ward_id,
        }

        # Process emotion data
        emotion_data = data.get("emotion")
        if emotion_data:
            emotion = emotion_data.get("emotion", "neutral")
            confidence = emotion_data.get("confidence", 0.0)
            intensity = emotion_data.get("intensity")

            logger.debug(
                f"[SensorStream] emotion={emotion} confidence={confidence:.2f} "
                f"intensity={intensity or 'N/A'}"
            )

            # Add to sensor_status
            sensor_status["emotion"] = {
                "status": self._determine_emotion_status(emotion, confidence),
                "value": emotion,
                "confidence": confidence,
            }

            # Forward to API for emotion buffering (enables report generation)
            asyncio.create_task(
                self._send_emotion_to_api(timestamp, emotion, confidence, intensity)
            )

            # Detect negative emotion alert
            result = self.sensor_detector.detect_emotion(emotion, confidence, intensity)
            if result.detected:
                self._handle_detection_result(timestamp, result)

        # Process audio data
        audio_data = data.get("audio")
        if audio_data:
            level = audio_data.get("level", 0.0)
            decibel = audio_data.get("decibel", 0.0)

            logger.info(
                f"[SensorStream] audio: level={level:.2f} decibel={decibel:.0f}dB"
            )

            # Add to sensor_status
            sensor_status["audio"] = {
                "status": self._determine_audio_status(level, decibel),
                "level": level,
                "decibel": decibel,
            }

            # Detect loud voice alert
            result = self.sensor_detector.detect_audio(level, decibel)
            if result.detected:
                self._handle_detection_result(timestamp, result)

        # Process motion data (new iOS fields)
        motion_data = data.get("motion")
        if motion_data:
            fall_risk = motion_data.get("fallRisk", 0.0)
            acceleration_magnitude = motion_data.get("accelerationMagnitude", 0.0)
            rotation_magnitude = motion_data.get("rotationMagnitude", 0.0)
            is_freefalling = motion_data.get("isFreefalling", False)

            logger.debug(
                f"[SensorStream] motion: fallRisk={fall_risk:.2f} "
                f"accMag={acceleration_magnitude:.2f} rotMag={rotation_magnitude:.2f} "
                f"freefall={is_freefalling}"
            )

            # Add to sensor_status
            sensor_status["motion"] = {
                "status": self._determine_motion_status(fall_risk),
                "fallRisk": fall_risk,
                "isFreefalling": is_freefalling,
            }

            # Detect device fall alert
            result = self.sensor_detector.detect_motion(
                fall_risk, acceleration_magnitude, rotation_magnitude, is_freefalling
            )
            if result.detected:
                self._handle_detection_result(timestamp, result)

        # Process face data (new iOS fields)
        face_data = data.get("face")
        if face_data:
            is_detected = face_data.get("isDetected", True)
            face_y = face_data.get("faceY", 0.0)
            y_delta = face_data.get("yDelta", 0.0)
            delta_time = face_data.get("deltaTime", 0.0)
            disappeared_duration = face_data.get("disappearedDuration", 0.0)

            logger.debug(
                f"[SensorStream] face: detected={is_detected} faceY={face_y:.2f} "
                f"yDelta={y_delta:.2f} deltaTime={delta_time:.2f} "
                f"disappearedDuration={disappeared_duration:.1f}"
            )

            # Add to sensor_status
            sensor_status["face"] = {
                "status": self._determine_face_status(
                    is_detected, y_delta, delta_time, disappeared_duration
                ),
                "isDetected": is_detected,
                "faceY": face_y,
            }

            # Detect person fall alert
            result = self.sensor_detector.detect_face(
                is_detected, face_y, y_delta, delta_time, disappeared_duration
            )
            if result.detected:
                self._handle_detection_result(timestamp, result)

        # Publish sensor_status to Web if any sensor data was present
        if any(key in sensor_status for key in ["emotion", "audio", "motion", "face"]):
            asyncio.create_task(self._publish_sensor_status(sensor_status))

    def _handle_detection_result(self, timestamp: int, result: DetectionResult) -> None:
        """Handle detection result from sensor detector.

        Creates internal CareAlert and processes it through normal flow.
        """
        if not result.detected or not result.alert_type:
            return

        # Map detector types to handler types
        alert_type_map = {
            DetectorAlertType.DEVICE_FALL: AlertType.DEVICE_FALL,
            DetectorAlertType.PERSON_FALL: AlertType.PERSON_FALL,
            DetectorAlertType.LOUD_VOICE: AlertType.LOUD_VOICE,
            DetectorAlertType.EMOTION: AlertType.EMOTION,
        }

        risk_level_map = {
            DetectorRiskLevel.NORMAL: RiskLevel.NORMAL,
            DetectorRiskLevel.CAUTION: RiskLevel.CAUTION,
            DetectorRiskLevel.CRITICAL: RiskLevel.CRITICAL,
        }

        alert_type = alert_type_map.get(result.alert_type)
        risk_level = risk_level_map.get(result.risk_level, RiskLevel.NORMAL)

        if not alert_type:
            return

        logger.info(
            f"[SensorStream] Detection triggered: {alert_type.value} "
            f"risk={risk_level.value} score={result.risk_score:.2f}"
        )

        # Create internal CareAlert
        alert = CareAlert(
            timestamp=timestamp,
            alert_type=alert_type,
            severity=AlertSeverity(result.severity) if result.severity in [e.value for e in AlertSeverity] else AlertSeverity.HIGH,
            payload=self._create_payload_from_result(alert_type, result),
            raw_payload=result.payload or {},
            risk_level=risk_level,
            risk_score=result.risk_score,
        )

        # Process through normal alert flow
        self._handle_alert(alert)

    def _create_payload_from_result(self, alert_type: AlertType, result: DetectionResult) -> PayloadType:
        """Create typed payload from detection result."""
        payload = result.payload or {}

        if alert_type == AlertType.DEVICE_FALL:
            return DeviceFallPayload(
                impact_magnitude=payload.get("impactMagnitude", 0),
                fall_type=DeviceFallType(payload.get("fallType", "impact")),
                freefall_duration=payload.get("freefallDuration"),
                max_rotation_rate=payload.get("maxRotationRate"),
            )
        elif alert_type == AlertType.PERSON_FALL:
            return PersonFallPayload(
                detection_type=PersonFallType(payload.get("detectionType", "rapid_descent")),
                face_y_delta=payload.get("faceYDelta"),
                delta_time=payload.get("deltaTime"),
                last_face_position=payload.get("lastFacePosition"),
            )
        elif alert_type == AlertType.LOUD_VOICE:
            return LoudVoicePayload(
                level=payload.get("level", 0),
                decibel=payload.get("decibel", 0),
                duration=payload.get("duration", 0),
                possible_cause=payload.get("possibleCause"),
            )
        elif alert_type == AlertType.EMOTION:
            emotion_str = payload.get("emotion", "neutral")
            try:
                emotion = EmotionType(emotion_str)
            except ValueError:
                emotion = EmotionType.NEUTRAL

            return EmotionPayload(
                emotion=emotion,
                confidence=payload.get("confidence", 0),
                intensity=payload.get("intensity"),
            )

        # Fallback
        return EmotionPayload(emotion=EmotionType.NEUTRAL, confidence=0)

    def _handle_help_request(self, data: dict) -> None:
        """Handle help_request from iOS (SOS button).

        Payload example:
        {
            "timestamp": 1705123456789,
            "wardId": "ward-uuid",
            "requestType": "help_needed"
        }

        Actions:
        1. TTS: "네, 무슨 일이세요? 말씀해주세요"
        2. API: send_care_alert with alertType='help_request', severity='high'
        3. iOS: notify via alert_response topic
        """
        timestamp = data.get("timestamp", 0) or int(time.time() * 1000)
        ward_id = data.get("wardId", self.ward_id)
        request_type = data.get("requestType", "help_needed")

        logger.info(
            f"[HelpRequest] Processing: ward={ward_id} requestType={request_type} "
            f"timestamp={timestamp}"
        )

        # 1. TTS response
        tts_message = "네, 무슨 일이세요? 말씀해주세요"
        asyncio.create_task(self.on_alert_response(tts_message))

        # 2. API: save help_request alert
        asyncio.create_task(
            self._publish_help_request(timestamp, ward_id, request_type, tts_message)
        )

        # 3. iOS notification via alert_response topic
        asyncio.create_task(
            self._notify_ios_help_request(timestamp, request_type, tts_message)
        )

    async def _publish_help_request(
        self,
        timestamp: int,
        ward_id: str,
        request_type: str,
        agent_response: str,
    ) -> None:
        """Send help_request to backend API."""
        alert_id = await send_care_alert(
            ward_id=ward_id,
            alert_type="help_request",
            severity="high",
            timestamp=timestamp,
            payload={"requestType": request_type},
            call_id=self.call_id,
            room_name=self.room.name,
            agent_response=agent_response,
            risk_level="critical",
            risk_score=1.0,
        )
        if alert_id:
            self.last_alert_id = alert_id
            logger.info(f"[HelpRequest] Stored alert_id={alert_id}")
        else:
            logger.warning(f"[HelpRequest] Failed to send to API: ward={ward_id}")

    async def _notify_ios_help_request(
        self,
        timestamp: int,
        request_type: str,
        agent_response: str,
    ) -> None:
        """Notify iOS about help_request acknowledgment."""
        try:
            payload = json.dumps({
                "alertType": "help_request",
                "severity": "high",
                "agentResponse": agent_response,
                "timestamp": timestamp,
                "detectionInfo": {
                    "type": "help_request",
                    "severity": "high",
                    "criteria": [
                        {
                            "name": "요청 유형",
                            "value": request_type,
                            "level": "high",
                        }
                    ],
                },
            })
            await self.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic=self.ALERT_RESPONSE_TOPIC,
            )
            logger.info(f"[HelpRequest] Notified iOS: help_request")
        except Exception as e:
            logger.error(f"[HelpRequest] Failed to notify iOS: {e}")

    async def _send_emotion_to_api(
        self,
        timestamp: int,
        emotion: str,
        confidence: float,
        intensity: Optional[float] = None,
    ) -> None:
        """Send emotion data to API for buffering.

        Uses the care-alerts endpoint with alertType='emotion' for buffering.
        This enables 10-minute aggregation and report generation.
        """
        success = await send_sensor_emotion(
            ward_id=self.ward_id,
            timestamp=timestamp,
            emotion=emotion,
            confidence=confidence,
            intensity=intensity,
        )
        if not success:
            logger.warning(f"[SensorStream] Failed to send emotion to API: ward={self.ward_id}")

    async def _publish_sensor_status(self, status_data: dict) -> None:
        """Publish sensor status to Web via DataChannel."""
        try:
            payload = json.dumps(status_data)
            await self.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=False,  # Real-time data uses unreliable for lower latency
                topic=self.SENSOR_STATUS_TOPIC,
            )
            logger.info(f"[SensorStatus] Published to Web: {status_data}")
        except Exception as e:
            logger.error(f"[SensorStatus] Failed to publish: {e}")

    def _determine_emotion_status(self, emotion: str, confidence: float) -> str:
        """Determine emotion status based on thresholds.

        - Negative emotions (sad, angry, fearful, disgusted) with confidence >= 0.8 -> critical
        - Negative emotions with confidence >= 0.6 -> caution
        - Otherwise -> normal
        """
        negative_emotions = {"sad", "angry", "fearful", "disgusted"}

        if emotion.lower() in negative_emotions:
            if confidence >= 0.8:
                return "critical"
            elif confidence >= 0.6:
                return "caution"

        return "normal"

    def _determine_audio_status(self, level: float, decibel: float) -> str:
        """Determine audio status based on thresholds.

        - level >= 0.8 or decibel >= 85 -> critical
        - level >= 0.6 or decibel >= 70 -> caution
        - Otherwise -> normal
        """
        if level >= 0.8 or decibel >= 85:
            return "critical"
        elif level >= 0.6 or decibel >= 70:
            return "caution"

        return "normal"

    def _determine_motion_status(self, fall_risk: float) -> str:
        """Determine motion status based on fallRisk thresholds.

        - fallRisk >= 0.7 -> critical
        - fallRisk >= 0.5 -> caution
        - Otherwise -> normal
        """
        if fall_risk >= 0.7:
            return "critical"
        elif fall_risk >= 0.5:
            return "caution"

        return "normal"

    def _determine_face_status(
        self,
        is_detected: bool,
        y_delta: float,
        delta_time: float,
        disappeared_duration: float,
    ) -> str:
        """Determine face status based on thresholds.

        - yDelta >= 0.25 && deltaTime <= 0.5 -> critical (rapid descent = 낙상)
        - Otherwise -> normal

        Note: 얼굴 미감지는 낙상이 아님 (걸어나감, 뒤돌아봄 등)
        """
        # 오직 급격한 하강만 critical (낙상 의심)
        if is_detected and y_delta >= 0.25 and delta_time <= 0.5 and delta_time > 0:
            return "critical"

        return "normal"
