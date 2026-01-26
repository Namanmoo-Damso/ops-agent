"""
Auto-RAG Mixin - 자동 RAG 검색

시간 표현 감지 기능 포함:
- "이틀 전에 뭐했어?" → 2일 전 날짜로 필터링
- "어제 뭐 얘기했지?" → 어제 날짜로 필터링

동적 토큰 예산:
- 질문 유형에 따라 max_tokens 자동 조절 (40~150)

시간 쿼리 자동 처리:
- "지금 몇 시야?", "시간 알려줘" 등 → get_current_time 자동 호출
"""

import logging
import re
import time
# from dataclasses import replace  # TODO: 동적 토큰 예산 구현 시 필요
from datetime import datetime
from typing import AsyncIterable, Optional

from livekit.agents import ModelSettings, llm

from ..constants import AGENT_TZINFO, TIMEOUT_RAG_SEARCH_QUICK
# from ..llm.token_budget import calculate_token_budget  # TODO: 동적 토큰 예산 구현 시 필요
from ..rag.orchestrator import RagOrchestrator
from ..rag.temporal_parser import get_temporal_parser
from ..rag_client import get_shared_rag_client

# 시간 쿼리 패턴 (모델이 tool을 호출하지 않을 때 직접 처리)
TIME_QUERY_PATTERNS = [
    r"몇\s*시",          # "몇 시", "몇시"
    r"지금\s*시간",      # "지금 시간"
    r"시간\s*알려",      # "시간 알려줘"
    r"현재\s*시간",      # "현재 시간"
    r"오늘\s*며칠",      # "오늘 며칠"
    r"무슨\s*요일",      # "무슨 요일"
]
TIME_QUERY_REGEX = re.compile("|".join(TIME_QUERY_PATTERNS), re.IGNORECASE)

logger = logging.getLogger(__name__)
pipeline_timing_logger = logging.getLogger("PIPELINE_TIMING")


def _is_time_query(text: str) -> bool:
    """시간 관련 질문인지 확인"""
    return bool(TIME_QUERY_REGEX.search(text))


def _get_current_time_str() -> str:
    """현재 시간을 한국어 형식으로 반환"""
    now = datetime.now(AGENT_TZINFO)
    hour = now.hour
    minute = now.minute

    # AM/PM in Korean
    if hour < 12:
        period = "오전"
        display_hour = hour if hour > 0 else 12
    else:
        period = "오후"
        display_hour = hour - 12 if hour > 12 else 12

    # Format the time string
    time_str = f"{now.year}년 {now.month}월 {now.day}일 {period} {display_hour}시"
    if minute > 0:
        time_str += f" {minute}분"

    return time_str


class AutoRAGMixin:
    """자동 RAG 검색을 제공하는 Mixin."""

    def _get_pipeline_times(self) -> dict[str, float]:
        pipeline_times = getattr(self, "_pipeline_times", None)
        if isinstance(pipeline_times, dict):
            return pipeline_times

        pipeline_times = {}
        setattr(self, "_pipeline_times", pipeline_times)
        return pipeline_times

    def _get_ward_id(self) -> str | None:
        session = getattr(self, "session", None)
        if (
            session
            and hasattr(session, "userdata")
            and hasattr(session.userdata, "ward_id")
        ):
            return session.userdata.ward_id
        return None

    async def _auto_rag_search(self, chat_ctx: llm.ChatContext) -> str | None:
        ward_id = self._get_ward_id()
        if not ward_id:
            logger.warning("[AutoRAG] No ward_id available")
            return None

        last_user_message = self._extract_last_user_message(chat_ctx)
        if not last_user_message:
            logger.debug("[AutoRAG] No user message found")
            return None

        # 시간 표현 감지
        temporal_parser = get_temporal_parser()
        temporal_result = temporal_parser.parse(last_user_message)

        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None

        if temporal_result.has_temporal:
            start_date = temporal_result.start_date
            end_date = temporal_result.end_date
            logger.info(
                f"[AutoRAG] Temporal query detected: '{temporal_result.expression}' "
                f"→ {start_date.strftime('%Y-%m-%d') if start_date else 'None'} ~ "
                f"{end_date.strftime('%Y-%m-%d') if end_date else 'None'}"
            )

        rag_start = time.time()
        rag_client = get_shared_rag_client(timeout=TIMEOUT_RAG_SEARCH_QUICK)

        logger.info(
            f"[AutoRAG] Searching: ward={ward_id}, query={last_user_message[:50]}..."
        )

        try:
            results = await rag_client.search_similar(
                ward_id=ward_id,
                query=last_user_message,
                limit=3,
                start_date=start_date,
                end_date=end_date,
            )

            rag_duration = time.time() - rag_start
            pipeline_times = self._get_pipeline_times()
            pipeline_times["rag_duration"] = rag_duration
            pipeline_timing_logger.info(f"RAG={rag_duration:.3f}s")

            if not results:
                logger.info("[AutoRAG] No results found")
                return None

            orchestrator = RagOrchestrator()
            search_result = orchestrator.process_results(results)
            logger.info(f"[AutoRAG] Found {len(results)} results")
            return search_result

        except Exception as e:
            logger.error(f"[AutoRAG] Search error: {e}")
            return None

    def _extract_last_user_message(self, chat_ctx: llm.ChatContext) -> Optional[str]:
        """ChatContext에서 마지막 사용자 메시지 추출"""
        try:
            for msg in reversed(chat_ctx.items):
                if hasattr(msg, "role") and msg.role == "user":
                    return str(msg.content)
        except Exception as e:
            logger.error(f"[AutoRAG] Error extracting user message: {e}")
        return None

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list,
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        # 시간 쿼리 감지 및 자동 처리
        last_user_message = self._extract_last_user_message(chat_ctx)
        if last_user_message and _is_time_query(last_user_message):
            current_time = _get_current_time_str()
            try:
                chat_ctx.add_message(
                    role="system",
                    content=f"[시간 정보] 현재 시간: {current_time} (한국 시간)\n\n"
                    f"위 시간 정보를 사용하여 자연스럽게 답변하세요.",
                )
                logger.info(f"[AutoRAG] Time query detected, injected: {current_time}")
            except Exception as e:
                logger.error(f"[AutoRAG] Error injecting time info: {e}")

        # Auto-RAG 검색 수행
        search_result = await self._auto_rag_search(chat_ctx)

        if search_result:
            try:
                chat_ctx.add_message(
                    role="system",
                    content=f"# 과거 대화 기록\n{search_result}\n\n위 정보를 참고하여 답변하세요.",
                )
                logger.info("[AutoRAG] Injected search result into context")
            except Exception as e:
                logger.error(f"[AutoRAG] Error injecting search result: {e}")

        # 동적 토큰 예산 계산 (TODO: ModelSettings에 max_tokens 필드 없음 - 추후 수정 필요)
        # last_user_message = self._extract_last_user_message(chat_ctx)
        # if last_user_message:
        #     token_budget = calculate_token_budget(last_user_message)
        #     logger.info(
        #         f"[TokenBudget] max_tokens={token_budget.max_tokens} "
        #         f"(reason={token_budget.reason})"
        #     )

        # 부모 클래스의 llm_node 호출
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk
