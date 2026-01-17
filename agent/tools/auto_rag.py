"""
Auto-RAG Mixin - 자동 RAG 검색
"""

import logging
import time
from typing import AsyncIterable

from livekit.agents import ModelSettings, llm

from ..constants import TIMEOUT_RAG_SEARCH_QUICK
from ..rag.orchestrator import RagOrchestrator
from ..rag_client import get_shared_rag_client

logger = logging.getLogger(__name__)
pipeline_timing_logger = logging.getLogger("PIPELINE_TIMING")


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

        last_user_message = None
        try:
            for msg in reversed(chat_ctx.items):
                if msg.role == "user":
                    last_user_message = str(msg.content)
                    break
        except Exception as e:
            logger.error(f"[AutoRAG] Error extracting user message: {e}")
            return None

        if not last_user_message:
            logger.debug("[AutoRAG] No user message found")
            return None

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

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list,
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
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

        # 부모 클래스의 llm_node 호출
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk
