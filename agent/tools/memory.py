"""
MemoryTool - search_memory 도구 정의 및 실행

역할: RAG 기반 과거 대화 검색 도구
- 도구 설명 Greedy화
- 광범위 키워드 지원
- RagOrchestrator 호출
"""

import logging
import time
from typing import Annotated

from livekit.agents import RunContext, function_tool
from pydantic import Field

from ..constants import TIMEOUT_RAG_SEARCH_QUICK
from ..rag_client import get_shared_rag_client
from ..rag.orchestrator import RagOrchestrator
from ..userdata import SessionUserdata

logger = logging.getLogger(__name__)
pipeline_timing_logger = logging.getLogger("PIPELINE_TIMING")


class MemoryToolMixin:
    """search_memory 도구를 제공하는 Mixin."""

    def _get_pipeline_times(self) -> dict[str, float]:
        """
        Ensure _pipeline_times exists even when AgentMetricsMixin is not mixed in.
        """
        pipeline_times = getattr(self, "_pipeline_times", None)
        if isinstance(pipeline_times, dict):
            return pipeline_times

        pipeline_times = {}
        setattr(self, "_pipeline_times", pipeline_times)
        return pipeline_times

    @function_tool
    async def search_memory(
        self,
        context: RunContext[SessionUserdata],
        query: Annotated[
            str,
            Field(
                description=(
                    "어르신의 모든 정보와 과거 기억 저장소입니다. "
                    "이름, 나이, 생일, 가족, 시장, 검진, 식사, 외출, 친구, 날씨, 기분, 약속 등 포함. "
                    "'내 이름', '뭐였더라?', '했었나?' 등 질문 시 무조건 사용하세요."
                )
            ),
        ],
    ) -> str:
        """
        어르신의 모든 과거 기억 저장소입니다.

        '어디', '무엇', '언제', '누구', '어떻게'에 대한 질문이 나오면
        당신의 답변보다 이 기록이 우선입니다.

        검색 대상 (매우 광범위!):
        - 가족: 손자, 아들, 딸, 며느리, 사위
        - 건강: 병원, 검진, 약, 무릎, 혈압
        - 일상: 시장, 장보기, 식사, 반찬, 외출, 산책
        - 감정/사건: 기분, 날씨, 명절, 생일, 기념일

        **조금이라도 의심되면 검색하세요. 검색 없이 '모르겠다'는 금지입니다.**

        Args:
            context: Run context with session userdata
            query: 검색 키워드 (예: '시장', '검진 결과', '크리스마스')

        Returns:
            과거 대화 기록 또는 검색 결과 없음 메시지
        """
        rag_start = time.time()
        rag_client = get_shared_rag_client(timeout=TIMEOUT_RAG_SEARCH_QUICK)
        ward_id = context.userdata.ward_id

        logger.info(f"RAG search (Parent-Child): ward={ward_id}, query={query}")

        try:
            results = await rag_client.search_similar(
                ward_id=ward_id,
                query=query,
                limit=3,
            )
            rag_duration = time.time() - rag_start
            pipeline_times = self._get_pipeline_times()
            pipeline_times["rag_duration"] = rag_duration
            pipeline_timing_logger.info(f"RAG={rag_duration:.3f}s")

            if not results:
                return "기록을 찾아봤지만 관련된 대화가 없었어요."

            orchestrator = RagOrchestrator()
            return orchestrator.process_results(results)

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return "기억을 불러오는 중 오류가 발생했어요. 잠시 후 다시 말씀해 주세요."
