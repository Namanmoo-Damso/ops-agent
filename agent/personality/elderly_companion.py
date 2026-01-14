"""Elderly Companion Agent - 어르신 돌봄 AI 에이전트."""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from constants import (
    AGENT_TZINFO,
    TIMEOUT_GREETING_FETCH,
    TIMEOUT_RAG_SEARCH_QUICK,
)
from livekit.agents import Agent, RunContext, function_tool
from pydantic import Field
from rag_client import (
    extract_result_text,
    extract_result_time_label,
    get_shared_rag_client,
)
from services.redis_pubsub import subscribe_to_greeting
from userdata import SessionUserdata

logger = logging.getLogger(__name__)


class CallDirection(str, Enum):
    """Call direction types."""

    INBOUND = "inbound"  # User calls agent
    OUTBOUND = "outbound"  # Agent calls user


# Greeting message constants
GREETING_OUTBOUND = "안녕하세요 어르신, 저 소담이에요."
GREETING_INBOUND = "네, 여보세요. 소담입니다."


class ElderlyCompanionAgent(Agent):
    """
    어르신을 위한 따뜻한 AI 동반자.

    한국어 존댓말을 사용하며, 어르신의 일상, 건강, 가족에 대해
    자연스럽게 대화합니다.
    """

    def __init__(self, ward_context: str = "", call_direction: str = "inbound") -> None:
        """
        Initialize the agent.

        Args:
            ward_context: Pre-fetched context about the ward (optional)
            call_direction: "inbound" or "outbound"
        """
        self.call_direction = call_direction
        super().__init__(
            instructions=self._build_instructions(ward_context, call_direction),
        )

    async def on_enter(self) -> None:
        """
        Called when agent enters the session - uses Push-based greeting approach.

        Push-based greeting strategy:
        1. Immediately say static greeting ("안녕하세요, 어르신" or "여보세요. 소담입니다")
        2. Subscribe to Redis Pub/Sub channel: greeting:ward:{wardId}
        3. When backend publishes personalized greeting, receive and say it

        This ensures zero-latency first response while receiving RAG-enhanced greeting
        when the backend is ready.
        """
        logger.info(
            f"ElderlyCompanionAgent entering with direction={self.call_direction}"
        )

        # Check if session is initialized
        if not hasattr(self, "session") or self.session is None:
            logger.error("Session not initialized in on_enter")
            return

        # Determine static greeting based on call direction
        if self.call_direction == CallDirection.OUTBOUND:
            fallback_greeting = GREETING_OUTBOUND
        else:
            fallback_greeting = GREETING_INBOUND

        # 📡 Hybrid Pull/Push approach for personalized greeting
        if hasattr(self.session, "userdata") and hasattr(
            self.session.userdata, "ward_id"
        ):
            ward_id = self.session.userdata.ward_id
            logger.info(
                f"[greeting] Checking for personalized greeting (ward={ward_id})"
            )

            greeting_received = asyncio.Event()
            start_time = time.monotonic()

            async def on_greeting_cached(personalized_greeting: str) -> None:
                """Called when greeting is found in cache (immediate)"""
                elapsed = time.monotonic() - start_time
                logger.info(
                    f"[greeting] Personalized greeting received from cache after {elapsed:.2f}s (ward={ward_id})"
                )
                greeting_received.set()
                # Say the full cached greeting immediately
                logger.info(
                    f"[greeting] Using cached greeting in full: {personalized_greeting[:50]}..."
                )
                self.session.say(personalized_greeting, allow_interruptions=False)

            async def on_greeting_pubsub(personalized_greeting: str) -> None:
                """Called when greeting arrives via Pub/Sub (delayed)"""
                elapsed = time.monotonic() - start_time
                logger.info(
                    f"[greeting] Personalized greeting received from Pub/Sub after {elapsed:.2f}s (ward={ward_id})"
                )
                greeting_received.set()
                # Process as before - extract additional content
                await self._on_greeting_received(personalized_greeting)

            # Try to get greeting (subscribe_to_greeting will check cache first)
            asyncio.create_task(
                self._log_greeting_timeout(greeting_received, start_time, ward_id)
            )
            asyncio.create_task(
                self._fetch_greeting_hybrid(
                    ward_id,
                    fallback_greeting,
                    on_greeting_cached,
                    on_greeting_pubsub,
                )
            )
        else:
            # No ward_id, just say static greeting
            logger.warning("Ward ID not available, using static greeting only")
            self.session.say(fallback_greeting, allow_interruptions=False)

    async def _fetch_greeting_hybrid(
        self,
        ward_id: str,
        fallback_greeting: str,
        on_cached,
        on_pubsub,
    ) -> None:
        """
        Fetch greeting using hybrid Pull/Push approach.

        This method wraps subscribe_to_greeting to handle the two different callbacks:
        - on_cached: Called if greeting is found in cache (immediate)
        - on_pubsub: Called if greeting arrives via Pub/Sub (after static greeting)
        """
        from services.redis_pubsub import get_redis_client

        try:
            client = await get_redis_client()
        except Exception as e:
            logger.error(f"[greeting] Redis client init failed: {e}")
            self.session.say(fallback_greeting, allow_interruptions=False)
            return
        if not client:
            logger.warning("[greeting] Redis unavailable, using static greeting")
            self.session.say(fallback_greeting, allow_interruptions=False)
            return

        # Check cache first
        cache_key = f"rag:greeting:ward:{ward_id}"
        try:
            cached_greeting = await client.get(cache_key)
            if cached_greeting:
                # Cache hit - use full greeting immediately
                await on_cached(cached_greeting)
                return
        except Exception as e:
            logger.error(f"[greeting] Error checking cache: {e}")

        # Cache miss - say static greeting first, then wait for Pub/Sub
        logger.info("[greeting] No cached greeting, saying static greeting first")
        try:
            self.session.say(fallback_greeting, allow_interruptions=False)
        except Exception as e:
            logger.error(f"[greeting] Failed to deliver static greeting: {e}")

        # Now subscribe for Pub/Sub updates
        await subscribe_to_greeting(
            ward_id,
            on_pubsub,
            timeout=TIMEOUT_GREETING_FETCH,
        )

    async def _on_greeting_received(self, personalized_greeting: str) -> None:
        """
        Callback invoked when personalized greeting is received via Redis Pub/Sub.

        This is called by the subscription when the backend publishes a greeting
        to the channel: greeting:ward:{wardId}

        Args:
            personalized_greeting: Full greeting text from backend RAG service
        """
        try:
            logger.info(
                f"[greeting] Received via Pub/Sub (length={len(personalized_greeting)})"
            )

            # Determine what static greeting was already said
            if self.call_direction == CallDirection.OUTBOUND:
                static_greeting = GREETING_OUTBOUND
            else:
                static_greeting = GREETING_INBOUND

            # Skip if it's identical to what we already said (after normalization)
            if self._normalize_greeting(
                personalized_greeting
            ) == self._normalize_greeting(static_greeting):
                logger.info("[greeting] Personalized greeting same as static, skipping")
                return

            # Extract additional content (remove static prefix if present)
            additional_content = self._extract_additional_content(
                personalized_greeting, static_greeting
            )

            if additional_content:
                logger.info(
                    f"[greeting] Adding personalized content: {additional_content[:50]}..."
                )
                # Remove leading dot and whitespace (e.g., ". 안녕하세요") using regex
                additional_content = additional_content.strip().lstrip(".").strip()
                # Say the additional personalized content
                self.session.say(additional_content, allow_interruptions=True)
            else:
                logger.info("[greeting] No additional content to add")

        except Exception as e:
            logger.error(
                f"[greeting] Error processing received greeting: {e}", exc_info=True
            )

    def _extract_additional_content(
        self, full_greeting: str, static_part: str
    ) -> Optional[str]:
        """
        Extract additional personalized content from full greeting.

        Tries multiple strategies:
        1. Remove exact static prefix
        2. Remove first sentence if it matches static greeting
        3. Return entire greeting if different from static

        Args:
            full_greeting: Complete greeting from RAG
            static_part: Static greeting prefix to remove

        Returns:
            Additional content to say, or None if nothing to add
        """
        full_greeting = self._normalize_greeting(full_greeting)
        static_part = self._normalize_greeting(static_part)

        # Strategy 1: Remove exact static prefix
        if full_greeting.startswith(static_part):
            additional = full_greeting[len(static_part) :].strip()
            if additional and len(additional) > 5:  # Meaningful content
                if self._normalize_greeting(additional) != static_part:
                    return additional

        # Strategy 2: Split by sentence and remove first if it's static greeting
        # Handle various sentence endings including Korean punctuation
        for delimiter in [". ", "。 ", ". ", ".\n", "! ", "? ", "！ ", "？ "]:
            if delimiter in full_greeting:
                sentences = full_greeting.split(delimiter, 1)
                if len(sentences) >= 2:
                    first_sentence = sentences[0].strip()
                    rest = sentences[1].strip()

                    # If first sentence matches static greeting, use rest
                    if (
                        first_sentence == static_part
                        or first_sentence + "." == static_part
                    ):
                        if (
                            rest
                            and len(rest) > 5
                            and self._normalize_greeting(rest) != static_part
                        ):
                            return rest

        # Strategy 3: If full greeting is completely different, use it
        if full_greeting != static_part and len(full_greeting) > len(static_part) + 10:
            return full_greeting

        return None

    def _normalize_greeting(self, text: str) -> str:
        """
        Normalize greeting strings for safer comparison.

        - Trim whitespace
        - Lowercase
        - Remove trailing common punctuation (., !, ?, ，, ？, ！, …)
        - Collapse double spaces
        """
        normalized = text.strip().lower()
        while normalized.endswith((".", "!", "?", "，", "？", "！", "…")):
            normalized = normalized[:-1].strip()
        normalized = " ".join(normalized.split())
        return normalized

    def _build_instructions(
        self, ward_context: str = "", call_direction: str = "inbound"
    ) -> str:
        """Build agent instructions with optional context and current time."""
        tz = AGENT_TZINFO
        local_now = datetime.now(tz)
        current_time_kst = local_now.strftime("%Y년 %m월 %d일 %H시 %M분")
        current_date_kst = local_now.strftime("%Y년 %m월 %d일")

        base = (
            "You are a warm, caring AI companion for elderly Korean users.\n"
            "Your name is '소담' (Sodam).\n\n"
            f"# 현재 시각 (한국 시간)\n"
            f"- 지금은 {current_time_kst} (KST) 입니다\n"
            f"- 오늘 날짜: {current_date_kst}\n\n"
            "# CRITICAL RULE: Language\n"
            "- User speaks: Korean (한국어)\n"
            "- You MUST respond: ONLY in Korean (한국어) using respectful 존댓말\n"
            "- NEVER respond in English - ALWAYS Korean\n"
            "- Example correct: '안녕하세요, 어르신'\n"
            "- Example WRONG: 'Hello' or any English\n"
            "- NEVER read special characters explicitely.\n\n"
        )

        memory_instruction = (
            "# Memory Usage & Temporal Awareness\n"
            "- When the user mentions family, health, past events, or personal topics, "
            "use the search_memory tool to recall previous conversations\n"
            "- Retrieved memories include date labels like '[날짜: 2026-01-12 14:30 KST]'\n"
            "- When available, a relative time hint like '[경과: 3일 전]' is provided; "
            "use it naturally (e.g., '어제 말씀하신 손자분...', '지난주에 병원 가셨다고 하셨죠?')\n"
            "- Use retrieved memories naturally without explicitly saying '기억을 검색했습니다'\n"
            "- If no relevant memory found, continue conversation naturally\n\n"
        )

        context_section = ""
        if ward_context:
            context_section = f"# 어르신 정보 (참고용)\n{ward_context}\n\n"

        output_rules = (
            "# Output rules\n"
            "- Use respectful Korean speech (존댓말) at all times\n"
            "- Keep responses brief: one to two sentences\n"
            "- Respond naturally to what they say\n"
            "- Be warm and caring in tone\n"
            "- Spell out numbers naturally: '세 시 반' not '3:30', '이천이십오년' not '2025년'\n"
            "- Never use emojis, special characters, or formatting\n"
            "- Avoid acronyms - say full words\n\n"
            "# Conversational flow\n"
            "- Listen more than you speak\n"
            "- Respond to their stories with empathy\n"
            "- Share relevant observations about wellbeing, meals, activities\n"
            "- Only ask questions when it naturally fits\n"
            "- Summarize key points when closing a topic\n\n"
            "# Tools\n"
            "- Use search_memory when user mentions family, health, or past events\n"
            "- Incorporate memories naturally without saying '기억을 검색했습니다'\n"
            "- If memory search fails, continue conversation gracefully\n\n"
            "# Handling interruptions\n"
            "- If interrupted, stop and listen\n"
            "- Acknowledge gracefully: '네, 말씀하세요'\n\n"
            "# Guardrails\n"
            "- For medical symptoms, recommend consulting a doctor: '의사 선생님께 여쭤보시는 게 좋겠어요'\n"
            "- Do not provide specific medication dosage advice\n"
            "- If emergency mentioned (chest pain, fall, etc.), suggest calling 119\n"
            "- Protect privacy - do not ask for sensitive personal information\n\n"
            "# Topics\n"
            "- Daily activities and meals\n"
            "- Health and feelings\n"
            "- Family and memories\n"
            "- Weather and seasons"
        )

        return base + memory_instruction + context_section + output_rules

    @function_tool
    async def search_memory(
        self,
        context: RunContext[SessionUserdata],
        query: Annotated[
            str,
            Field(
                description="검색할 키워드나 주제 (예: '손자', '병원', '약', '가족')"
            ),
        ],
    ) -> str:
        """
        어르신과의 과거 대화 기록을 검색합니다 (Parent-Child 구조 활용).

        어르신이 이전에 언급한 내용(가족 이름, 건강 상태, 취미, 과거 이야기 등)을
        기억해서 자연스럽게 대화해야 할 때 사용합니다.

        Parent-Child 구조를 통해 더 넓은 문맥을 제공하면서도 관련성 높은 부분을 강조합니다.

        Args:
            context: Run context with session userdata
            query: Search query (e.g., '손자', '병원', '약')

        Returns:
            Retrieved memory context with parent context or message if not found
        """
        rag_client = get_shared_rag_client(timeout=TIMEOUT_RAG_SEARCH_QUICK)
        ward_id = context.userdata.ward_id

        logger.info(f"RAG search (Parent-Child): ward={ward_id}, query={query}")

        try:
            results = await rag_client.search_similar(
                ward_id=ward_id,
                query=query,
                limit=3,  # Top 3 for quick recall
            )

            if not results:
                return "관련된 과거 대화가 없습니다."

            # Format results into context with parent context and temporal information
            context_parts = []
            for result in results:
                text = extract_result_text(result)
                relative_time = extract_result_time_label(result)

                # API already filtered by SIMILARITY_THRESHOLD, so trust all returned results
                if text:
                    if relative_time:
                        context_parts.append(f"{text}\n[경과: {relative_time}]")
                    else:
                        context_parts.append(text)

            if not context_parts:
                return "관련된 과거 대화가 없습니다."

            context_text = "\n\n".join(context_parts)
            return f"어르신과의 과거 대화에서 찾은 정보:\n{context_text}"

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return "기억 검색 중 오류가 발생했습니다."

    async def _log_greeting_timeout(
        self,
        greeting_received: asyncio.Event,
        start_time: float,
        ward_id: str,
    ) -> None:
        await asyncio.sleep(TIMEOUT_GREETING_FETCH)
        if greeting_received.is_set():
            return
        elapsed = time.monotonic() - start_time
        logger.warning(
            f"[greeting] No personalized greeting within {elapsed:.2f}s (ward={ward_id}); "
            "using static greeting only"
        )
