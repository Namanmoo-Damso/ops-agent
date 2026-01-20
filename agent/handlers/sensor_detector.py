"""
SensorDetector - 센서 데이터 기반 위험 감지

역할: sensor_stream 데이터를 분석하여 caution/critical 알림 생성
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    CRITICAL = "critical"


class AlertType(str, Enum):
    DEVICE_FALL = "device_fall"
    PERSON_FALL = "person_fall"
    LOUD_VOICE = "loud_voice"
    EMOTION = "emotion"


@dataclass
class DetectionResult:
    """감지 결과"""
    detected: bool
    alert_type: Optional[AlertType] = None
    risk_level: RiskLevel = RiskLevel.NORMAL
    risk_score: float = 0.0
    severity: str = "low"  # low, medium, high, critical
    payload: Optional[dict] = None
    message: Optional[str] = None


# ============================================================================
# 임계값 설정 (caution / critical)
# ============================================================================

class Thresholds:
    """센서별 임계값 설정"""

    # Emotion (시연용으로 confidence 낮춤)
    EMOTION_CAUTION_CONFIDENCE = 0.3   # 주의: 30% 이상
    EMOTION_CRITICAL_CONFIDENCE = 0.7  # 경고: 70% 이상
    EMOTION_NEGATIVE = {"sad", "angry", "fearful", "disgusted"}

    # Audio (loud_voice)
    AUDIO_CAUTION_DECIBEL = 70.0       # 주의: 70dB 이상
    AUDIO_CRITICAL_DECIBEL = 85.0      # 경고: 85dB 이상
    AUDIO_CAUTION_LEVEL = 0.6          # 주의: level 0.6 이상
    AUDIO_CRITICAL_LEVEL = 0.8         # 경고: level 0.8 이상
    AUDIO_CAUTION_DURATION = 1.0       # 주의: 1초 이상 지속
    AUDIO_CRITICAL_DURATION = 2.0      # 경고: 2초 이상 지속

    # Motion (device_fall) - fallRisk 기반 (iOS가 계산해서 전송)
    # fallRisk 범위: 0.0 ~ 1.0
    MOTION_FALLRISK_NORMAL_MAX = 0.49      # normal: 0.0 ~ 0.49
    MOTION_FALLRISK_CAUTION_MAX = 0.69     # caution: 0.50 ~ 0.69
    # critical: 0.70 ~ 1.0

    # Face (person_fall) - iOS가 계산해서 전송하는 값 활용
    FACE_YDELTA_THRESHOLD = 0.25          # yDelta >= 0.25 → person_fall
    FACE_DELTATIME_THRESHOLD = 0.5        # deltaTime <= 0.5초 이내
    FACE_DISAPPEAR_THRESHOLD = 5.0        # 얼굴 미감지 5초 이상 → person_fall


class SensorDetector:
    """센서 데이터 기반 위험 감지기"""

    def __init__(self):
        # Audio state tracking
        self._audio_loud_start: Optional[float] = None
        self._last_audio_level: float = 0.0
        self._last_audio_decibel: float = 0.0

        # Cooldown to prevent duplicate alerts (per alert type)
        self._last_alert_time: dict[AlertType, float] = {}
        self.ALERT_COOLDOWN = 10.0  # 같은 타입 알림 간 최소 간격 (초)

    def _is_on_cooldown(self, alert_type: AlertType) -> bool:
        """알림 쿨다운 체크"""
        last_time = self._last_alert_time.get(alert_type, 0)
        return (time.time() - last_time) < self.ALERT_COOLDOWN

    def _record_alert(self, alert_type: AlertType) -> None:
        """알림 시간 기록"""
        self._last_alert_time[alert_type] = time.time()

    # ========================================================================
    # Emotion Detection
    # ========================================================================

    def detect_emotion(self, emotion: str, confidence: float, intensity: Optional[float] = None) -> DetectionResult:
        """감정 기반 위험 감지"""

        # 부정적 감정이 아니면 정상
        if emotion not in Thresholds.EMOTION_NEGATIVE:
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        # 쿨다운 체크
        if self._is_on_cooldown(AlertType.EMOTION):
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        risk_level = RiskLevel.NORMAL
        severity = "low"
        risk_score = confidence

        if confidence >= Thresholds.EMOTION_CRITICAL_CONFIDENCE:
            risk_level = RiskLevel.CRITICAL
            severity = "high"
        elif confidence >= Thresholds.EMOTION_CAUTION_CONFIDENCE:
            risk_level = RiskLevel.CAUTION
            severity = "medium"
        else:
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        self._record_alert(AlertType.EMOTION)

        logger.info(
            f"[Detector] EMOTION detected: {emotion} conf={confidence:.2f} "
            f"risk={risk_level.value}"
        )

        return DetectionResult(
            detected=True,
            alert_type=AlertType.EMOTION,
            risk_level=risk_level,
            risk_score=risk_score,
            severity=severity,
            payload={
                "emotion": emotion,
                "confidence": confidence,
                "intensity": intensity,
            },
            message=f"부정적 감정 감지: {emotion} (신뢰도 {confidence*100:.0f}%)"
        )

    # ========================================================================
    # Audio Detection (Loud Voice)
    # ========================================================================

    def detect_audio(self, level: float, decibel: float) -> DetectionResult:
        """오디오 기반 위험 감지 (큰 소리)"""

        now = time.time()
        is_loud = False

        # 임계값 초과 체크
        if decibel >= Thresholds.AUDIO_CAUTION_DECIBEL or level >= Thresholds.AUDIO_CAUTION_LEVEL:
            is_loud = True

        if is_loud:
            if self._audio_loud_start is None:
                self._audio_loud_start = now

            duration = now - self._audio_loud_start
            self._last_audio_level = level
            self._last_audio_decibel = decibel

            # 쿨다운 체크
            if self._is_on_cooldown(AlertType.LOUD_VOICE):
                return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

            risk_level = RiskLevel.NORMAL
            severity = "low"

            # Critical 조건
            if (decibel >= Thresholds.AUDIO_CRITICAL_DECIBEL or level >= Thresholds.AUDIO_CRITICAL_LEVEL) \
                    and duration >= Thresholds.AUDIO_CRITICAL_DURATION:
                risk_level = RiskLevel.CRITICAL
                severity = "critical"
            # Caution 조건
            elif duration >= Thresholds.AUDIO_CAUTION_DURATION:
                risk_level = RiskLevel.CAUTION
                severity = "high"
            else:
                return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

            self._record_alert(AlertType.LOUD_VOICE)
            self._audio_loud_start = None  # Reset

            logger.info(
                f"[Detector] LOUD_VOICE detected: {decibel:.0f}dB level={level:.2f} "
                f"duration={duration:.1f}s risk={risk_level.value}"
            )

            return DetectionResult(
                detected=True,
                alert_type=AlertType.LOUD_VOICE,
                risk_level=risk_level,
                risk_score=min(1.0, level),
                severity=severity,
                payload={
                    "level": level,
                    "decibel": decibel,
                    "duration": duration,
                },
                message=f"큰 소리 감지: {decibel:.0f}dB ({duration:.1f}초 지속)"
            )
        else:
            # 소리가 작아지면 리셋
            self._audio_loud_start = None

        return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

    # ========================================================================
    # Motion Detection (Device Fall) - fallRisk 기반
    # ========================================================================

    def detect_motion(
        self,
        fall_risk: float,
        acceleration_magnitude: float,
        rotation_magnitude: float,
        is_freefalling: bool,
    ) -> DetectionResult:
        """모션 기반 위험 감지 (기기 낙상)

        Args:
            fall_risk: iOS가 계산한 낙상 위험도 (0.0 ~ 1.0)
                - normal: 0.0 ~ 0.49
                - caution: 0.50 ~ 0.69
                - critical: 0.70 ~ 1.0
            acceleration_magnitude: 가속도 크기 (g)
            rotation_magnitude: 회전 크기 (rad/s)
            is_freefalling: 자유낙하 여부
        """

        # False positive 필터: isFreefalling=True이지만 가속도가 매우 낮으면 무시
        # 폰이 테이블에 평평하게 놓이면 가속도가 ~0g가 되어 iOS가 freefall로 오판
        # 진짜 낙하는 짧은 순간(< 1초) 후 충격이 옴, 계속 0g 유지 = 테이블 위 정지 상태
        if is_freefalling and acceleration_magnitude < 0.1:
            logger.debug(
                f"[Detector] Ignoring false positive freefall: "
                f"accMag={acceleration_magnitude:.4f}g (too low, likely phone on table)"
            )
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        # 쿨다운 체크
        if self._is_on_cooldown(AlertType.DEVICE_FALL):
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        # fallRisk 기준으로 판단
        risk_level = RiskLevel.NORMAL
        severity = "low"

        if fall_risk >= 0.70:
            risk_level = RiskLevel.CRITICAL
            severity = "critical"
        elif fall_risk >= 0.50:
            risk_level = RiskLevel.CAUTION
            severity = "medium"  # caution은 medium (high 아님)
        else:
            # normal 범위 (0.0 ~ 0.49) - 알림 없음
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        self._record_alert(AlertType.DEVICE_FALL)

        fall_type = "freefall_impact" if is_freefalling else "impact"

        logger.info(
            f"[Detector] DEVICE_FALL detected: fallRisk={fall_risk:.2f} "
            f"accMag={acceleration_magnitude:.2f}g rotMag={rotation_magnitude:.2f} "
            f"freefall={is_freefalling} risk={risk_level.value}"
        )

        return DetectionResult(
            detected=True,
            alert_type=AlertType.DEVICE_FALL,
            risk_level=risk_level,
            risk_score=fall_risk,
            severity=severity,
            payload={
                "fallRisk": fall_risk,
                "impactMagnitude": acceleration_magnitude,
                "rotationMagnitude": rotation_magnitude,
                "fallType": fall_type,
                "isFreefalling": is_freefalling,
            },
            message=f"기기 낙상 감지: 위험도 {fall_risk*100:.0f}%"
        )

    # ========================================================================
    # Face Detection (Person Fall) - iOS 계산 값 활용
    # ========================================================================

    def detect_face(
        self,
        is_detected: bool,
        face_y: float,
        y_delta: float,
        delta_time: float,
        disappeared_duration: float,
    ) -> DetectionResult:
        """얼굴 기반 위험 감지 (사람 낙상)

        Args:
            is_detected: 얼굴 감지 여부
            face_y: 현재 얼굴 Y 위치
            y_delta: 최근 0.5초간 Y 변화량 (양수=하강)
            delta_time: Y 변화 측정 시간 (초)
            disappeared_duration: 얼굴 미감지 지속 시간 (초) - 현재 미사용

        판단 기준:
            - yDelta >= 0.25 && deltaTime <= 0.5 → person_fall (급격한 하강)

        Note:
            얼굴 미감지는 낙상으로 판단하지 않음
            (걸어나감, 뒤돌아봄, 조명 문제, 카메라 가림 등 다양한 이유)
        """

        # 쿨다운 체크
        if self._is_on_cooldown(AlertType.PERSON_FALL):
            return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

        # 얼굴 미감지는 낙상이 아님 (걸어나감, 뒤돌아봄, 조명 문제 등)
        # 오직 "급격한 하강"만 낙상으로 판단

        # yDelta >= 0.25 && deltaTime <= 0.5 → person_fall (급격한 하강)
        if (
            is_detected
            and y_delta >= Thresholds.FACE_YDELTA_THRESHOLD
            and delta_time <= Thresholds.FACE_DELTATIME_THRESHOLD
            and delta_time > 0
        ):
            self._record_alert(AlertType.PERSON_FALL)

            logger.info(
                f"[Detector] PERSON_FALL detected: rapid_descent "
                f"yDelta={y_delta:.2f} deltaTime={delta_time:.2f}s"
            )

            return DetectionResult(
                detected=True,
                alert_type=AlertType.PERSON_FALL,
                risk_level=RiskLevel.CRITICAL,
                risk_score=min(1.0, y_delta / 0.5),
                severity="critical",
                payload={
                    "detectionType": "rapid_descent",
                    "faceY": face_y,
                    "yDelta": y_delta,
                    "deltaTime": delta_time,
                },
                message=f"급격한 하강 감지: Y변화 {y_delta*100:.0f}% ({delta_time:.2f}초)"
            )

        return DetectionResult(detected=False, risk_level=RiskLevel.NORMAL)

    def reset(self) -> None:
        """상태 초기화"""
        self._audio_loud_start = None
        self._last_alert_time.clear()
        logger.info("[Detector] State reset")
