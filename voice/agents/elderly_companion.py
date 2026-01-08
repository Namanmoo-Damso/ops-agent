"""Elderly Companion Agent - 어르신 돌봄 AI 에이전트."""
import logging
from enum import Enum
from typing import Annotated

from livekit.agents import Agent, RunContext, function_tool
from pydantic import Field

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
        """Called when agent enters the session - greets based on call direction."""
        logger.info(f"ElderlyCompanionAgent entering with direction={self.call_direction}")

        # Check if session is initialized
        if not hasattr(self, 'session') or self.session is None:
            logger.error("Session not initialized in on_enter")
            return

        # Use different greetings based on call direction
        if self.call_direction == CallDirection.OUTBOUND:
            greeting = GREETING_OUTBOUND
        else:
            greeting = GREETING_INBOUND

        self.session.say(greeting, allow_interruptions=False)

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
            "- Be warm and caring in tone\n\n"
            "# Conversational flow\n"
            "- Listen more than you speak\n"
            "- Respond to their stories with empathy\n"
            "- Share relevant observations about wellbeing, meals, activities\n"
            "- Only ask questions when it naturally fits\n\n"
            "# Handling interruptions\n"
            "- If interrupted, stop and listen\n"
            "- Acknowledge gracefully: '네, 말씀하세요'\n\n"
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
        import httpx
        import os

        api_base = os.getenv("API_BASE_URL")
        if not api_base:
            logger.error("API_BASE_URL not configured")
            return "시스템 설정 오류가 발생했습니다."

        api_token = os.getenv("API_INTERNAL_TOKEN")
        ward_id = context.userdata.ward_id

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        logger.info(f"RAG search: ward={ward_id}, query={query}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{api_base}/v1/rag/search",
                    json={
                        "wardId": ward_id,
                        "query": query,
                        "topK": 5,
                    },
                    headers=headers,
                    timeout=3.0,
                )

                if response.status_code != 200:
                    logger.warning(f"RAG search failed: {response.status_code}")
                    return "관련 기억을 찾을 수 없습니다."

                data = response.json()
                context_text = data.get("context", "")

                if not context_text:
                    return "관련된 과거 대화가 없습니다."

                return f"어르신과의 과거 대화에서 찾은 정보:\n{context_text}"

        except httpx.TimeoutException:
            logger.warning("RAG search timeout")
            return "기억 검색 시간이 초과되었습니다."
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return "기억 검색 중 오류가 발생했습니다."
