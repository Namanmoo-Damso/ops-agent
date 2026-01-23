"""
GreetingManager - 인사말 관리 모듈

[전체 로직 요약]
1. 초기화: 호출 방향(Inbound/Outbound)에 따라 기본 정적 인사말을 결정합니다.
2. 진입 시 (on_enter):
   - ward_id가 없으면 즉시 정적 인사말을 송출합니다.
   - ward_id가 있으면 하이브리드(Pull/Push) 모드로 전환합니다.
3. 하이브리드 Fetch:
   - (Pull) Redis 캐시에 개인화된 인사말이 있는지 확인합니다. 있다면 즉시 송출하고 종료합니다.
   - (Push) 캐시가 없으면 1.5초간 실시간 인사말 생성을 기다립니다.
   - 1.5초 내에 생성되면 개인화 인사말을 전체 송출합니다.
   - 1.5초가 넘어가면 정적 인사말을 송출하고, 이후 도착하는 개인화 인사말은 무시합니다.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Protocol, Union

from ..constants import TIMEOUT_GREETING_FETCH
from ..services.redis_pubsub import get_redis_client, subscribe_to_greeting

logger = logging.getLogger(__name__)


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
        """Agent enters session - Pull-cached or Push-pubsub greeting approach."""
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

        # 1. Ward ID가 없는 경우: 즉시 정적 인사말 송출 후 종료
        if not ward_id:
            logger.warning("Ward ID not available, using static greeting only")
            session.say(self._static_greeting, allow_interruptions=False)
            return

        # 2. 캐시 확인: 캐시가 있으면 즉시 송출하고 종료
        try:
            if client := await get_redis_client():
                cache_key = f"rag:greeting:ward:{ward_id}"
                if cached_greeting := await client.get(cache_key):
                    logger.info(f"[greeting] Cache hit (ward={ward_id})")
                    session.say(cached_greeting, allow_interruptions=False)
                    return
        except Exception as e:
            logger.error(f"[greeting] Redis cache check failed: {e}")

        # 3. 캐시가 없는 경우: 1초간 Pub/Sub 메시지를 기다려봄
        logger.info(
            "[greeting] Cache miss, waiting 1s for Pub/Sub before static fallback"
        )

        start_time = time.monotonic()
        greeting_received = asyncio.Event()
        is_static_spoken = False

        async def on_pubsub(personalized_greeting: str) -> None:
            nonlocal is_static_spoken
            greeting_received.set()

            if is_static_spoken:
                logger.info(
                    "[greeting] Pub/Sub arrived too late, ignoring as static greeting already spoken"
                )
                return

            # 1초 이내에 도착한 경우 -> 전체 개인화 인사말 송출
            logger.info(
                f"[greeting] Pub/Sub fast arrival ({time.monotonic() - start_time:.2f}s)"
            )
            session.say(personalized_greeting, allow_interruptions=False)

        # 구독 시작
        asyncio.create_task(
            subscribe_to_greeting(ward_id, on_pubsub, timeout=TIMEOUT_GREETING_FETCH)
        )
        asyncio.create_task(
            self._log_greeting_timeout(greeting_received, start_time, ward_id)
        )

        try:
            # 0.5초 동안 Pub/Sub 메시지 대기 (1.5초→0.5초: 첫 인사 지연 단축)
            await asyncio.wait_for(greeting_received.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            # 0.5초 내에 안 오면 정적 인사말 송출
            is_static_spoken = True
            logger.info("[greeting] 0.5s wait timeout, saying static greeting first")
            session.say(self._static_greeting, allow_interruptions=False)

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
