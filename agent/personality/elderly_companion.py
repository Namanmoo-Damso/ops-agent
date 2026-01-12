"""Elderly Companion Agent - 어르신 돌봄 AI 에이전트."""
import asyncio
import logging
from enum import Enum
from typing import Annotated, Optional

from livekit.agents import Agent, RunContext, function_tool
from pydantic import Field

from constants import TIMEOUT_RAG_SEARCH_QUICK, TIMEOUT_GREETING_FETCH
from rag_client import get_shared_rag_client
from services.redis_pubsub import subscribe_to_greeting
from userdata import SessionUserdata

logger = logging.getLogger(__name__)


class CallDirection(str, Enum):
    """Call direction types."""
    INBOUND = "inbound"   # User calls agent
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
        logger.info(f"ElderlyCompanionAgent entering with direction={self.call_direction}")

        # Check if session is initialized
        if not hasattr(self, 'session') or self.session is None:
            logger.error("Session not initialized in on_enter")
            return

        # Determine static greeting based on call direction
        if self.call_direction == CallDirection.OUTBOUND:
            fallback_greeting = GREETING_OUTBOUND
        else:
            fallback_greeting = GREETING_INBOUND

        # 🚀 STEP 1: Immediately say static greeting (zero latency)
        logger.info(f"💬 Saying static greeting immediately (direction={self.call_direction})")
        self.session.say(fallback_greeting, allow_interruptions=False)

        # 📡 STEP 2: Subscribe to greeting channel (non-blocking, Push-based)
        if hasattr(self.session, 'userdata') and hasattr(self.session.userdata, 'ward_id'):
            ward_id = self.session.userdata.ward_id
            # Launch background task to subscribe and wait for greeting
            asyncio.create_task(
                subscribe_to_greeting(
                  ward_id,
                  self._on_greeting_received,
                  timeout=TIMEOUT_GREETING_FETCH,
                )
            )
        else:
            logger.warning("Ward ID not available, skipping personalized greeting subscription")

    async def _on_greeting_received(self, personalized_greeting: str) -> None:
        """
        Callback invoked when personalized greeting is received via Redis Pub/Sub.

        This is called by the subscription when the backend publishes a greeting
        to the channel: greeting:ward:{wardId}

        Args:
            personalized_greeting: Full greeting text from backend RAG service
        """
        try:
            logger.info(f"🎯 Greeting received via Pub/Sub (length={len(personalized_greeting)})")

            # Determine what static greeting was already said
            if self.call_direction == CallDirection.OUTBOUND:
                static_greeting = GREETING_OUTBOUND
            else:
                static_greeting = GREETING_INBOUND

            # Skip if it's identical to what we already said (after normalization)
            if self._normalize_greeting(personalized_greeting) == self._normalize_greeting(static_greeting):
                logger.info("⚠️  Personalized greeting same as static, skipping")
                return

            # Extract additional content (remove static prefix if present)
            additional_content = self._extract_additional_content(personalized_greeting, static_greeting)

            if additional_content:
                logger.info(f"✨ Adding personalized content: {additional_content[:50]}...")
                # Say the additional personalized content
                self.session.say(additional_content, allow_interruptions=True)
            else:
                logger.info("📋 No additional content to add")

        except Exception as e:
            logger.error(f"❌ Error processing received greeting: {e}", exc_info=True)

    def _extract_additional_content(self, full_greeting: str, static_part: str) -> Optional[str]:
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
            additional = full_greeting[len(static_part):].strip()
            if additional and len(additional) > 5:  # Meaningful content
                return additional

        # Strategy 2: Split by sentence and remove first if it's static greeting
        # Handle various sentence endings: '. ', '。', '. '
        for delimiter in ['. ', '。 ', '. ', '.\n']:
            if delimiter in full_greeting:
                sentences = full_greeting.split(delimiter, 1)
                if len(sentences) >= 2:
                    first_sentence = sentences[0].strip()
                    rest = sentences[1].strip()

                    # If first sentence matches static greeting, use rest
                    if first_sentence == static_part or first_sentence + '.' == static_part:
                        if rest and len(rest) > 5:
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
        - Remove trailing common punctuation
        - Collapse double spaces
        """
        normalized = text.strip().lower()
        while normalized.endswith(('.', '!', '?')):
            normalized = normalized[:-1].strip()
        normalized = " ".join(normalized.split())
        return normalized

    def _build_instructions(self, ward_context: str = "", call_direction: str = "inbound") -> str:
        """Build agent instructions with optional context."""
        base = (
            "You are a warm, caring AI companion for elderly Korean users.\n"
            "Your name is '소담' (Sodam).\n\n"
            "# CRITICAL RULE: Language\n"
            "- User speaks: Korean (한국어)\n"
            "- You MUST respond: ONLY in Korean (한국어) using respectful 존댓말\n"
            "- NEVER respond in English - ALWAYS Korean\n"
            "- Example correct: '안녕하세요, 어르신'\n"
            "- Example WRONG: 'Hello' or any English\n\n"
        )

        memory_instruction = (
            "# Memory Usage\n"
            "- When the user mentions family, health, past events, or personal topics, "
            "use the search_memory tool to recall previous conversations\n"
            "- Use retrieved memories naturally without explicitly saying '기억을 검색했습니다'\n"
            "- If no relevant memory found, continue conversation naturally\n\n"
        )

        context_section = ""
        if ward_context:
            context_section = (
                "# 어르신 정보 (참고용)\n"
                f"{ward_context}\n\n"
            )

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
            Field(description="검색할 키워드나 주제 (예: '손자', '병원', '약', '가족')"),
        ],
    ) -> str:
        """
        어르신과의 과거 대화 기록을 검색합니다.

        어르신이 이전에 언급한 내용(가족 이름, 건강 상태, 취미, 과거 이야기 등)을
        기억해서 자연스럽게 대화해야 할 때 사용합니다.

        Args:
            context: Run context with session userdata
            query: Search query (e.g., '손자', '병원', '약')

        Returns:
            Retrieved memory context or message if not found
        """
        rag_client = get_shared_rag_client(timeout=TIMEOUT_RAG_SEARCH_QUICK)
        ward_id = context.userdata.ward_id

        logger.info(f"RAG search: ward={ward_id}, query={query}")

        try:
            results = await rag_client.search_similar(
                ward_id=ward_id,
                query=query,
                limit=3,  # Top 3 for quick recall
            )

            if not results:
                return "관련된 과거 대화가 없습니다."

            # Format results into context
            context_parts = []
            for result in results:
                text = result.get("text", "")
                similarity = result.get("similarity", 0)

                # Only include results with reasonable similarity (>0.5)
                if similarity > 0.5 and text:
                    context_parts.append(text)

            if not context_parts:
                return "관련된 과거 대화가 없습니다."

            context_text = "\n\n".join(context_parts)
            return f"어르신과의 과거 대화에서 찾은 정보:\n{context_text}"

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return "기억 검색 중 오류가 발생했습니다."
