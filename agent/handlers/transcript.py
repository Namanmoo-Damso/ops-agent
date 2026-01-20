"""Transcript handling - publishing and broadcasting."""
import asyncio
import json
import logging
import time
from typing import Set, Callable, Optional, Awaitable

from ..services.redis_pubsub import publish_transcript, store_transcript_direct
from ..services.api_client import send_care_alert
from .negative_keyword_detector import detect_keywords, KeywordDetectionResult, RiskLevel, CATEGORY_LABELS

logger = logging.getLogger(__name__)

# 카테고리별 LLM 응답 메시지
KEYWORD_RESPONSE_MESSAGES: dict[str, str] = {
    # 통증 관련
    "pain_head": "어르신, 머리가 아프신가요? 어디가 어떻게 아프신지 말씀해 주세요.",
    "pain_chest": "어르신, 가슴이 아프신 건가요? 정확히 어디가 어떻게 아프신지 알려주세요.",
    "pain_abdominal": "어르신, 배가 아프신가요? 언제부터 아프셨어요?",
    "pain_back": "어르신, 허리나 등이 불편하신가요? 어디가 아프신지 말씀해 주세요.",
    "pain_joint": "어르신, 관절이 아프신가요? 어느 부위가 아프신지 알려주세요.",
    "pain_general": "어르신, 어디가 아프세요? 괜찮으신가요?",

    # 증상 관련
    "symptom_dizziness": "어르신, 어지러우신가요? 지금 안전한 곳에 계신가요?",
    "symptom_breathing": "어르신, 숨쉬기가 힘드신가요? 천천히 숨을 쉬어보세요.",
    "symptom_nausea": "어르신, 속이 안 좋으신가요? 언제부터 그러셨어요?",
    "symptom_weakness": "어르신, 기운이 없으신가요? 오늘 식사는 하셨나요?",
    "symptom_numbness": "어르신, 어디가 저리시거나 감각이 없으신가요? 어느 부위인지 말씀해 주세요.",
    "symptom_fever": "어르신, 열이 나시나요? 체온을 재보셨나요?",
    "symptom_sleep": "어르신, 잠이 잘 안 오시나요? 언제부터 그러셨어요?",

    # 정서적 고통
    "emotional_depression": "어르신, 마음이 힘드신가요? 무슨 일이 있으셨어요?",
    "emotional_anxiety": "어르신, 걱정되는 일이 있으신가요? 무엇이 불안하신지 말씀해 주세요.",
    "emotional_loneliness": "어르신, 외로우셨군요. 제가 함께 이야기 나눠드릴게요.",
    "emotional_fear": "어르신, 무서우신가요? 무엇이 두려우신지 말씀해 주세요.",
    "emotional_anger": "어르신, 화가 나셨군요. 무슨 일이 있으셨어요?",

    # 자해/자살 관련 (최고 위험)
    "suicide_ideation": "어르신, 지금 많이 힘드시죠. 제가 옆에 있을게요. 무슨 일이 있으셨는지 말씀해 주세요.",
    "self_harm": "어르신, 괜찮으세요? 지금 어디 다치신 곳은 없으신가요?",

    # 긴급 상황
    "emergency_fall": "어르신, 넘어지셨나요? 다친 곳은 없으신가요?",
    "emergency_help": "어르신, 도움이 필요하신가요? 무슨 일이신지 말씀해 주세요.",
    "emergency_accident": "어르신, 다치셨나요? 어디가 아프신지 말씀해 주세요.",

    # 건강 걱정
    "health_concern": "어르신, 건강이 걱정되시는 부분이 있으신가요?",
}

# 기본 응답 메시지 (카테고리가 없거나 매핑되지 않은 경우)
DEFAULT_RESPONSE_MESSAGE = "어르신, 괜찮으세요? 무슨 일이신지 말씀해 주세요."


class TranscriptHandler:
    """Handles transcript publishing and broadcasting."""

    # Topic for speech keyword alerts (to Web)
    SPEECH_ALERT_TOPIC = "speech_alert"

    def __init__(
        self,
        call_id: str,
        room,
        ward_id: Optional[str] = None,
        on_keyword_detected: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.call_id = call_id
        self.room = room
        self.ward_id = ward_id
        self.on_keyword_detected = on_keyword_detected
        self._background_tasks: Set[asyncio.Task] = set()

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """Create a task and track it to prevent memory leaks."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def publish_transcript_event(self, speaker: str, text: str):
        """Publish transcript event to Redis with fallback."""
        success = await publish_transcript(self.call_id, speaker, text)
        if not success:
            await store_transcript_direct(self.call_id, speaker, text)

    async def broadcast_to_room(self, role: str, text: str):
        """Broadcast transcript to room via data packet."""
        try:
            payload = json.dumps({
                "type": "transcript",
                "role": role,
                "text": text,
                "timestamp": int(time.time() * 1000)
            })
            await self.room.local_participant.publish_data(
                payload,
                reliable=True,
                topic="transcript",
            )
        except Exception as e:
            logger.error(f"Failed to broadcast transcript: {e}")

    async def _publish_speech_alert(self, detection: KeywordDetectionResult, text: str):
        """Publish speech keyword alert to Web via DataChannel and API."""
        timestamp = int(time.time() * 1000)
        matched_keywords = list(set(kw for _, kw in detection.all_matches))

        # 1. Web DataChannel로 speech_alert 발송 (사이드바 업데이트용)
        try:
            payload = json.dumps({
                "timestamp": timestamp,
                "wardId": self.ward_id,
                "speech": {
                    "status": detection.risk_level.value,  # caution or critical
                    "category": detection.category.value if detection.category else None,
                    "categoryLabel": detection.category_label,
                    "matchedKeyword": detection.matched_keyword,
                    "matchedKeywords": matched_keywords,
                    "text": text,
                    "matchCount": len(detection.all_matches),
                },
            })
            await self.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic=self.SPEECH_ALERT_TOPIC,
            )
            logger.info(
                f"[SpeechAlert] Published to Web: category={detection.category_label}, "
                f"keyword='{detection.matched_keyword}', risk={detection.risk_level.value}"
            )
        except Exception as e:
            logger.error(f"Failed to publish speech alert to Web: {e}")

        # 2. API로 care_alert 발송 (room metadata 업데이트 및 테두리 색상 변경용)
        alert_id = None
        try:
            severity = "high" if detection.risk_level.value == "critical" else "medium"
            alert_id = await send_care_alert(
                ward_id=self.ward_id,
                alert_type="speech_keyword",  # 발화 키워드 전용 타입 (emotion 버퍼링 우회)
                severity=severity,
                timestamp=timestamp,
                payload={
                    "category": detection.category.value if detection.category else None,
                    "categoryLabel": detection.category_label,
                    "matchedKeyword": detection.matched_keyword,
                    "matchedKeywords": matched_keywords,
                    "text": text,
                    "source": "speech_keyword",  # 발화 키워드 감지임을 표시
                },
                call_id=self.call_id,
                room_name=self.room.name,
                agent_response=None,  # 키워드 감지는 TTS 응답 없음
                risk_level=detection.risk_level.value,
                risk_score=min(1.0, len(detection.all_matches) * 0.3),
            )
            if alert_id:
                logger.info(f"[SpeechAlert] Sent to API: alertId={alert_id}")
            else:
                logger.warning(f"[SpeechAlert] Failed to send to API")
        except Exception as e:
            logger.error(f"Failed to send speech alert to API: {e}")

        # 3. iOS에 alert_response 토픽으로 알림 발송 (iOS 앱에서 알림 표시)
        if alert_id:
            try:
                ios_payload = json.dumps({
                    "alertId": alert_id,
                    "alertType": "speech_keyword",
                })
                await self.room.local_participant.publish_data(
                    ios_payload.encode("utf-8"),
                    reliable=True,
                    topic="alert_response",
                )
                logger.info(
                    f"[SpeechAlert] Published to iOS: alertId={alert_id}, alertType=speech_keyword"
                )
            except Exception as e:
                logger.error(f"Failed to publish alert to iOS: {e}")

    def handle_user_transcript(self, ev, takeover_active: bool = False):
        """Handle user transcript event."""
        if takeover_active:
            return
        if ev.is_final:
            logger.debug(f"User transcript: {ev.transcript}")
            self._create_tracked_task(self.publish_transcript_event("user", ev.transcript))
            self._create_tracked_task(self.broadcast_to_room("user", ev.transcript))

            # 발화 키워드 감지 및 Web으로 경고 발송
            detection = detect_keywords(ev.transcript)
            if detection.detected:
                self._create_tracked_task(self._publish_speech_alert(detection, ev.transcript))

                # LLM 응답 트리거 (caution 이상일 때만)
                if self.on_keyword_detected and detection.risk_level in (RiskLevel.CAUTION, RiskLevel.CRITICAL):
                    # 카테고리에 맞는 응답 메시지 선택
                    category_key = detection.category.value if detection.category else None
                    response_message = KEYWORD_RESPONSE_MESSAGES.get(category_key, DEFAULT_RESPONSE_MESSAGE)
                    logger.info(
                        f"[SpeechKeyword] Triggering LLM response: category={category_key}, "
                        f"risk={detection.risk_level.value}, message='{response_message[:30]}...'"
                    )
                    self._create_tracked_task(self.on_keyword_detected(response_message))

    def handle_agent_speech(self, ev):

        """Handle agent speech committed event."""
        if hasattr(ev, 'content') and ev.content:
            logger.debug(f"Agent response: {ev.content}")
            self._create_tracked_task(self.publish_transcript_event("agent", ev.content))

    def handle_conversation_item(self, ev):
        """Handle conversation item added event."""
        try:
            item = ev.item if hasattr(ev, 'item') else ev
            if hasattr(item, 'role') and item.role == 'assistant':
                content = self._extract_content(item)
                if content and content.strip():
                    self._create_tracked_task(self.publish_transcript_event("agent", content))
                    self._create_tracked_task(self.broadcast_to_room("agent", content))
        except Exception as e:
            logger.error(f"Error in conversation_item_added handler: {e}")

    def _extract_content(self, item) -> str:
        """Extract text content from conversation item."""
        if hasattr(item, 'text') and item.text:
            return item.text
        elif hasattr(item, 'content'):
            raw_content = item.content
            if isinstance(raw_content, str):
                return raw_content
            elif isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif hasattr(part, 'text'):
                        text_parts.append(part.text)
                    elif hasattr(part, '__str__'):
                        text_parts.append(str(part))
                return ' '.join(text_parts)
            else:
                return str(raw_content)
        return ""

    async def wait_for_pending_publications(self, timeout: float = 10.0):
        """Wait for all pending transcript publication tasks to finish."""
        if not self._background_tasks:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        logger.info(f"Waiting for {len(self._background_tasks)} transcript publication task(s) to finish")

        pending = self._background_tasks.copy()

        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning("Timeout while waiting for transcript publications to complete")
                break

            done, pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Drop completed tasks from tracker to avoid re-waiting
            self._background_tasks.difference_update(done)

            # Refresh pending set with any still-running tracked tasks
            pending = self._background_tasks.copy()

        if self._background_tasks:
            # ensure any remaining tasks are awaited to avoid warnings
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
