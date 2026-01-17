"""
GreetingManager - 인사말 관리 모듈

[전체 로직 요약]
1. 초기화: 호출 방향(Inbound/Outbound)에 따라 기본 정적 인사말을 결정합니다.
2. 진입 시 (on_enter):
   - ward_id가 없으면 즉시 정적 인사말을 송출합니다.
   - ward_id가 있으면 하이브리드(Pull/Push) 모드로 전환합니다.
3. 하이브리드 Fetch:
   - (Pull) Redis 캐시에 개인화된 인사말이 있는지 확인합니다. 있다면 즉시 송출하고 종료합니다.
   - (Push) 캐시가 없으면 정적 인사말을 먼저 송출하고, Redis Pub/Sub을 구독하여 실시간 생성을 기다립니다.
4. 추가 콘텐츠 처리 (_on_greeting_received):
   - Pub/Sub을 통해 뒤늦게 생성된 인사말이 도착하면, 이미 송출된 정적 인사말과 비교합니다.
   - 정적 인사말 부분을 제외한 '새로 추가된 내용'만 추출하여 추가로 송출합니다.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional, Protocol, Union

from ..constants import TIMEOUT_GREETING_FETCH
from ..services.redis_pubsub import subscribe_to_greeting

logger = logging.getLogger(__name__)

MIN_ADDITIONAL_CONTENT_LENGTH = 6
MIN_FULL_GREETING_EXTRA_CHARS = 10


class SessionUserdataLike(Protocol):
    """Minimal session userdata shape needed for greeting flow."""

    ward_id: str


class AgentSessionLike(Protocol):
    """Minimal session interface used by GreetingManagerMixin."""

    userdata: SessionUserdataLike

    def say(self, text: str, allow_interruptions: bool) -> None: ...


class CallDirection(Enum):
    """Call direction types."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


# Greeting message constants
GREETING_OUTBOUND = "안녕하세요 어르신, 저 소담이에요."
GREETING_INBOUND = "네, 여보세요. 소담이에요."


class GreetingManagerMixin:
    """인사말 관리를 위한 Mixin 클래스."""

    session: Optional[AgentSessionLike]

    def __init__(
        self,
        *args,
        call_direction: Union[CallDirection, str] = CallDirection.INBOUND,
        **kwargs,
    ):
        """call_direction 및 ward_context 초기화."""
        self.ward_context = kwargs.pop("ward_context", None)
        super().__init__(*args, **kwargs)

        self.call_direction = (
            call_direction
            if isinstance(call_direction, CallDirection)
            else CallDirection(call_direction)
        )

    @property
    def _static_greeting(self) -> str:
        """호출 방향에 따른 기본 인사말 반환."""
        return (
            GREETING_OUTBOUND
            if self.call_direction == CallDirection.OUTBOUND
            else GREETING_INBOUND
        )

    async def on_enter(self) -> None:
        """Agent enters session - Push-based greeting approach."""
        logger.info("VoiceAgent entering with direction=%s", self.call_direction.value)

        session = getattr(self, "session", None)
        if not session:
            logger.error("Session not initialized in on_enter")
            return

        ward_id = (
            getattr(session.userdata, "ward_id", None)
            if hasattr(session, "userdata")
            else None
        )
        if not ward_id:
            logger.warning("Ward ID not available, using static greeting only")
            session.say(self._static_greeting, allow_interruptions=False)
            return

        logger.info(f"[greeting] Checking for personalized greeting (ward={ward_id})")
        greeting_received = asyncio.Event()
        start_time = time.monotonic()

        async def on_cached(personalized_greeting: str) -> None:
            elapsed = time.monotonic() - start_time
            logger.info(f"[greeting] From cache after {elapsed:.2f}s")
            greeting_received.set()
            session.say(personalized_greeting, allow_interruptions=False)

        async def on_pubsub(personalized_greeting: str) -> None:
            elapsed = time.monotonic() - start_time
            logger.info(f"[greeting] From Pub/Sub after {elapsed:.2f}s")
            greeting_received.set()
            await self._on_greeting_received(personalized_greeting)

        asyncio.create_task(
            self._log_greeting_timeout(greeting_received, start_time, ward_id)
        )
        asyncio.create_task(self._fetch_greeting_hybrid(ward_id, on_cached, on_pubsub))

    async def _fetch_greeting_hybrid(
        self,
        ward_id: str,
        on_cached: Callable,
        on_pubsub: Callable,
    ) -> None:
        """Fetch greeting using hybrid Pull/Push approach."""
        from ..services.redis_pubsub import get_redis_client

        session = getattr(self, "session", None)
        if not session:
            return

        try:
            if client := await get_redis_client():
                cache_key = f"rag:greeting:ward:{ward_id}"
                if cached_greeting := await client.get(cache_key):
                    await on_cached(cached_greeting)
                    return
        except Exception as e:
            logger.error(f"[greeting] Redis error: {e}")

        logger.info("[greeting] No cached greeting, saying static greeting first")
        session.say(self._static_greeting, allow_interruptions=False)
        await subscribe_to_greeting(ward_id, on_pubsub, timeout=TIMEOUT_GREETING_FETCH)

    async def _on_greeting_received(self, personalized_greeting: str) -> None:
        """Process personalized greeting from Pub/Sub."""
        try:
            session = getattr(self, "session", None)
            if not session:
                return

            static_greeting = self._static_greeting
            if self._normalize_greeting(
                personalized_greeting
            ) == self._normalize_greeting(static_greeting):
                logger.info("[greeting] Received same as static, skipping")
                return

            additional = self._extract_additional_content(
                personalized_greeting, static_greeting
            )
            if additional:
                additional = additional.strip().lstrip(".").strip()
                logger.info(f"[greeting] Adding: {additional[:50]}...")
                session.say(additional, allow_interruptions=True)
        except Exception as e:
            logger.error(f"[greeting] Error: {e}", exc_info=True)

    def _extract_additional_content(
        self, full_greeting: str, static_part: str
    ) -> Optional[str]:
        """Extract additional personalized content from full greeting."""
        full_norm = self._normalize_greeting(full_greeting)
        static_norm = self._normalize_greeting(static_part)

        if full_norm == static_norm:
            return None

        # 1. Prefix match
        if full_norm.startswith(static_norm):
            additional = full_norm[len(static_norm) :].strip()
            if len(additional) >= MIN_ADDITIONAL_CONTENT_LENGTH:
                return additional

        # 2. Sentence split match
        for delimiter in [". ", "。 ", "! ", "? ", "！ ", "？ "]:
            if delimiter in full_norm:
                parts = full_norm.split(delimiter, 1)
                if self._normalize_greeting(parts[0]) == static_norm:
                    rest = parts[1].strip()
                    if len(rest) >= MIN_ADDITIONAL_CONTENT_LENGTH:
                        return rest

        # 3. Fallback: length based
        if len(full_norm) > len(static_norm) + MIN_FULL_GREETING_EXTRA_CHARS:
            return full_norm

        return None

    def _normalize_greeting(self, text: str) -> str:
        """Normalize greeting strings for comparison."""
        normalized = text.strip().lower()
        while normalized.endswith((".", "!", "?", "，", "？", "！", "…")):
            normalized = normalized[:-1].strip()
        return " ".join(normalized.split())

    async def _log_greeting_timeout(
        self, greeting_received: asyncio.Event, start_time: float, ward_id: str
    ) -> None:
        """Log warning if greeting not received within timeout."""
        await asyncio.sleep(TIMEOUT_GREETING_FETCH)
        if not greeting_received.is_set():
            elapsed = time.monotonic() - start_time
            logger.warning(
                f"[greeting] No personalized greeting within {elapsed:.2f}s (ward={ward_id})"
            )
