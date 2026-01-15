"""
GreetingManager - 인사말 관리

역할: Redis Pub/Sub 연동, 인사말 추출/정규화 및 하이브리드 인사말 로직
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional, Protocol

from constants import TIMEOUT_GREETING_FETCH
from services.redis_pubsub import subscribe_to_greeting

logger = logging.getLogger(__name__)

class SessionUserdataLike(Protocol):
    """Minimal session userdata shape needed for greeting flow."""

    ward_id: str


class AgentSessionLike(Protocol):
    """Minimal session interface used by GreetingManagerMixin."""

    userdata: SessionUserdataLike

    def say(self, text: str, allow_interruptions: bool) -> None:
        ...


class CallDirection(Enum):
    """Call direction types."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


# Greeting message constants
GREETING_OUTBOUND = "안녕하세요 어르신, 저 소담이에요."
GREETING_INBOUND = "네, 여보세요. 소담입니다."


class GreetingManagerMixin:
    """인사말 관리를 위한 Mixin 클래스."""

    session: Optional[AgentSessionLike]

    def __init__(self, *args, call_direction: str = "inbound", **kwargs):
        """call_direction 초기화."""
        super().__init__(*args, **kwargs)
        self.call_direction = (
            CallDirection.OUTBOUND
            if call_direction == "outbound"
            else CallDirection.INBOUND
        )

    async def on_enter(self) -> None:
        """Agent enters session - Push-based greeting approach."""
        logger.info(f"ElderlyCompanionAgent entering with direction={self.call_direction}")

        session = getattr(self, "session", None)
        if session is None:
            logger.error("Session not initialized in on_enter")
            return

        fallback_greeting = (
            GREETING_OUTBOUND
            if self.call_direction == CallDirection.OUTBOUND
            else GREETING_INBOUND
        )

        if hasattr(session, "userdata") and hasattr(session.userdata, "ward_id"):
            ward_id = session.userdata.ward_id
            logger.info(f"[greeting] Checking for personalized greeting (ward={ward_id})")

            greeting_received = asyncio.Event()
            start_time = time.monotonic()

            async def on_greeting_cached(personalized_greeting: str) -> None:
                elapsed = time.monotonic() - start_time
                logger.info(f"[greeting] From cache after {elapsed:.2f}s (ward={ward_id})")
                greeting_received.set()
                session.say(personalized_greeting, allow_interruptions=False)

            async def on_greeting_pubsub(personalized_greeting: str) -> None:
                elapsed = time.monotonic() - start_time
                logger.info(f"[greeting] From Pub/Sub after {elapsed:.2f}s (ward={ward_id})")
                greeting_received.set()
                await self._on_greeting_received(personalized_greeting)

            asyncio.create_task(self._log_greeting_timeout(greeting_received, start_time, ward_id))
            asyncio.create_task(
                self._fetch_greeting_hybrid(ward_id, fallback_greeting, on_greeting_cached, on_greeting_pubsub)
            )
        else:
            logger.warning("Ward ID not available, using static greeting only")
            session.say(fallback_greeting, allow_interruptions=False)

    async def _fetch_greeting_hybrid(
        self, ward_id: str, fallback_greeting: str, on_cached: Callable, on_pubsub: Callable
    ) -> None:
        """Fetch greeting using hybrid Pull/Push approach."""
        from services.redis_pubsub import get_redis_client

        session = getattr(self, "session", None)
        if session is None:
            logger.error("Session not initialized in greeting fetch")
            return

        try:
            client = await get_redis_client()
        except Exception as e:
            logger.error(f"[greeting] Redis client init failed: {e}")
            session.say(fallback_greeting, allow_interruptions=False)
            return

        if not client:
            logger.warning("[greeting] Redis unavailable, using static greeting")
            session.say(fallback_greeting, allow_interruptions=False)
            return

        cache_key = f"rag:greeting:ward:{ward_id}"
        try:
            cached_greeting = await client.get(cache_key)
            if cached_greeting:
                await on_cached(cached_greeting)
                return
        except Exception as e:
            logger.error(f"[greeting] Error checking cache: {e}")

        logger.info("[greeting] No cached greeting, saying static greeting first")
        try:
            session.say(fallback_greeting, allow_interruptions=False)
        except Exception as e:
            logger.error(f"[greeting] Failed to deliver static greeting: {e}")

        await subscribe_to_greeting(ward_id, on_pubsub, timeout=TIMEOUT_GREETING_FETCH)

    async def _on_greeting_received(self, personalized_greeting: str) -> None:
        """Process personalized greeting from Pub/Sub."""
        try:
            logger.info(f"[greeting] Received via Pub/Sub (length={len(personalized_greeting)})")
            session = getattr(self, "session", None)
            if session is None:
                logger.error("Session not initialized when greeting received")
                return

            static_greeting = (
                GREETING_OUTBOUND
                if self.call_direction == CallDirection.OUTBOUND
                else GREETING_INBOUND
            )

            if self._normalize_greeting(personalized_greeting) == self._normalize_greeting(static_greeting):
                logger.info("[greeting] Same as static, skipping")
                return

            additional_content = self._extract_additional_content(personalized_greeting, static_greeting)
            if additional_content:
                additional_content = additional_content.strip().lstrip(".").strip()
                logger.info(f"[greeting] Adding: {additional_content[:50]}...")
                session.say(additional_content, allow_interruptions=True)
            else:
                logger.info("[greeting] No additional content to add")
        except Exception as e:
            logger.error(f"[greeting] Error: {e}", exc_info=True)

    def _extract_additional_content(self, full_greeting: str, static_part: str) -> Optional[str]:
        """Extract additional personalized content from full greeting."""
        full_norm = self._normalize_greeting(full_greeting)
        static_norm = self._normalize_greeting(static_part)

        if full_norm.startswith(static_norm):
            additional = full_norm[len(static_norm):].strip()
            if additional and len(additional) > 5 and self._normalize_greeting(additional) != static_norm:
                return additional

        for delimiter in [". ", "。 ", ". ", ".\n", "! ", "? ", "！ ", "？ "]:
            if delimiter in full_norm:
                sentences = full_norm.split(delimiter, 1)
                if len(sentences) >= 2:
                    first_sentence = sentences[0].strip()
                    rest = sentences[1].strip()
                    if first_sentence == static_norm or first_sentence + "." == static_norm:
                        if rest and len(rest) > 5 and self._normalize_greeting(rest) != static_norm:
                            return rest

        if full_norm != static_norm and len(full_norm) > len(static_norm) + 10:
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
