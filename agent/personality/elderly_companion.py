"""
ElderlyCompanionAgent - 어르신 돌봄 AI 에이전트

메인 엔트리포인트: 모듈들을 조합하여 Agent 클래스 정의
- AgentMetricsMixin: 파이프라인 타이밍
- GreetingManagerMixin: 인사말 관리
- PersonaManagerMixin: 페르소나/프롬프트
- MemoryToolMixin: search_memory 도구
"""

import logging

from livekit.agents import Agent

from .agent_metrics import AgentMetricsMixin
from .greeting_manager import GreetingManagerMixin, CallDirection
from .persona_manager import PersonaManagerMixin
from .memory_tool import MemoryToolMixin

logger = logging.getLogger(__name__)


class ElderlyCompanionAgent(
    AgentMetricsMixin,
    GreetingManagerMixin,
    PersonaManagerMixin,
    MemoryToolMixin,
    Agent,
):
    """
    어르신을 위한 따뜻한 AI 동반자.

    한국어 존댓말을 사용하며, 어르신의 일상, 건강, 가족에 대해
    자연스럽게 대화합니다.

    구성 요소:
    - AgentMetricsMixin: STT/LLM/TTS 타이밍 측정
    - GreetingManagerMixin: Redis Pub/Sub 인사말 관리
    - PersonaManagerMixin: 시스템 프롬프트 생성
    - MemoryToolMixin: RAG 기반 기억 검색 도구
    """

    def __init__(self, ward_context: str = "", call_direction: str = "inbound"):
        """
        Initialize the agent.

        Args:
            ward_context: Pre-fetched context about the ward (optional)
            call_direction: "inbound" or "outbound"
        """
        # Build instructions with persona
        instructions = self._build_instructions(ward_context, call_direction)

        # Initialize Agent with instructions
        super().__init__(instructions=instructions)

        # Store for later use
        self._ward_context = ward_context
        self.call_direction = (
            CallDirection.OUTBOUND
            if call_direction == "outbound"
            else CallDirection.INBOUND
        )

        logger.info(
            f"ElderlyCompanionAgent initialized: "
            f"direction={call_direction}, context_len={len(ward_context)}"
        )
