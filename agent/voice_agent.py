"""
VoiceAgent - 어르신 돌봄 AI 에이전트

메인 엔트리포인트: 모듈들을 조합하여 Agent 클래스 정의
- AutoRAGMixin: 자동 과거 대화 검색
- PipelineTimerMixin: 파이프라인 타이밍
- GreetingManagerMixin: 인사말 관리
- Instructions Built via PromptBuilder (Persona Integrated)
"""

import logging
from typing import Union

from livekit.agents import Agent

from .pipeline_timer import PipelineTimerMixin
from .prompts.greeting import CallDirection, GreetingManagerMixin
from .tools.auto_rag import AutoRAGMixin
from .tools.time import TimeToolMixin
from .tools.ward_info import WardInfoToolMixin

logger = logging.getLogger(__name__)


class VoiceAgent(
    AutoRAGMixin,
    TimeToolMixin,
    WardInfoToolMixin,
    PipelineTimerMixin,
    GreetingManagerMixin,
    Agent,
):
    """
    어르신을 위한 따뜻한 AI 동반자.

    한국어 존댓말을 사용하며, 어르신의 일상, 건강, 가족에 대해
    자연스럽게 대화합니다.

    구성 요소:
    - AutoRAGMixin: 자동 과거 대화 검색 및 컨텍스트 주입
    - PipelineTimerMixin: STT/LLM/TTS 타이밍 측정
    - GreetingManagerMixin: Redis Pub/Sub 인사말 관리
    """

    def __init__(
        self,
        ward_context: str = "",
        call_direction: Union[CallDirection, str] = CallDirection.INBOUND,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            ward_context: Pre-fetched context about the ward (optional)
            call_direction: "inbound" or "outbound"
            latitude: Ward's current location latitude (for weather tools)
            longitude: Ward's current location longitude (for weather tools)

        Note:
            ward_context, latitude, longitude are stored in SessionUserdata
            and accessed via tools (get_ward_info, get_ward_location).
            This keeps the system prompt 100% static for vLLM prefix cache.
        """
        self._ward_context = ward_context
        self._prompt_builder = None

        # Build static instructions (no dynamic ward data injected)
        instructions = self._build_instructions()

        # Initialize via mixin chain
        super().__init__(
            ward_context=ward_context,
            call_direction=call_direction,
            instructions=instructions,
        )

        logger.info(
            f"VoiceAgent initialized: "
            f"direction={self.call_direction.value}, context_len={len(ward_context)}"
        )

    def _build_instructions(self) -> str:
        """
        Build agent instructions using YAML-based prompt builder.

        Returns static prompt without ward-specific data.
        Ward info is accessed via tools (get_ward_info, get_ward_location).
        """
        # Lazy import to avoid circular dependency
        if self._prompt_builder is None:
            from .prompts.builder import PromptBuilder

            self._prompt_builder = PromptBuilder(template_name="sodam")

        return self._prompt_builder.build()
