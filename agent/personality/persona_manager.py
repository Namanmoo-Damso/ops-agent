"""
PersonaManager - 페르소나/시스템 프롬프트 생성

역할: YAML 기반 프롬프트 빌더를 사용한 시스템 프롬프트 생성
"""

import logging
from typing import Union

from ..prompts.greeting import CallDirection

logger = logging.getLogger(__name__)


class PersonaManagerMixin:
    """페르소나 및 시스템 프롬프트 관리 Mixin."""

    def __init__(self, *args, ward_context: str = "", **kwargs):
        """ward_context 초기화."""
        super().__init__(*args, **kwargs)
        self._ward_context = ward_context
        self._prompt_builder = None

    def _build_instructions(
        self,
        ward_context: str = "",
        call_direction: Union[CallDirection, str] = CallDirection.INBOUND,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> str:
        """Build agent instructions using YAML-based prompt builder."""
        # Lazy import to avoid circular dependency
        if self._prompt_builder is None:
            from ..prompts.builder import PromptBuilder

            self._prompt_builder = PromptBuilder(template_name="sodam")

        return self._prompt_builder.build(
            ward_context=ward_context,
            latitude=latitude,
            longitude=longitude,
        )
