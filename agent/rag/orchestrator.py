"""
RagOrchestrator - RAG 오케스트레이션

역할: 검색 → 필터링 → 포맷팅 → 프롬프트 주입 과정 조율
"""

import logging
import time
from datetime import datetime
from typing import Any, Optional

from .threshold_filter import SearchThresholdFilter
from .context_formatter import ContextFormatter
from .prompt_template import AgentPromptTemplate

logger = logging.getLogger(__name__)


class RagOrchestrator:
    """RAG 검색 결과 처리 오케스트레이터."""

    def __init__(
        self,
        threshold_filter: Optional[SearchThresholdFilter] = None,
        context_formatter: Optional[ContextFormatter] = None,
    ):
        """
        Args:
            threshold_filter: 결과 필터링 (기본 생성)
            context_formatter: 결과 포맷팅 (기본 생성)
        """
        self.filter = threshold_filter or SearchThresholdFilter()
        self.formatter = context_formatter or ContextFormatter()

    def process_results(
        self,
        results: list[dict[str, Any]],
        current_time: Optional[datetime] = None,
    ) -> str:
        """
        RAG 검색 결과를 LLM에 주입할 형식으로 처리.

        Args:
            results: RAG 검색 결과
            current_time: 현재 시간

        Returns:
            LLM에 주입할 컨텍스트 문자열
        """
        start = time.time()

        # Step 1: 필터링
        logger.info(f"[Orchestrator] Raw results: {len(results)} items")
        filtered = self.filter.filter_results(results)
        logger.info(f"[Orchestrator] After filter: {len(filtered)} items")

        # Step 2: 포맷팅
        formatted = self.formatter.format_for_llm(filtered, current_time)
        logger.info(f"[Orchestrator] Formatted context: {len(formatted)} chars")

        # Step 3: 프롬프트 블록 생성
        context_block = AgentPromptTemplate.build_memory_context_block(formatted)
        
        elapsed = time.time() - start
        logger.info(
            f"[Orchestrator] Processing complete in {elapsed:.3f}s: "
            f"{len(results)} -> {len(filtered)} results, {len(context_block)} chars"
        )

        # 최종 주입될 텍스트 미리보기
        preview = context_block[:100].replace("\n", " ")
        logger.info(f"[Orchestrator] Context preview: {preview}...")

        return context_block

    @staticmethod
    def get_enhanced_instructions() -> str:
        """강화된 시스템 프롬프트 지침 반환."""
        return (
            AgentPromptTemplate.get_memory_instruction()
            + AgentPromptTemplate.get_temporal_awareness_instruction()
        )
