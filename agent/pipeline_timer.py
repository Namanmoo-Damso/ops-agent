"""
PipelineTimer - 파이프라인 타이밍 측정

역할: STT/LLM/TTS 노드의 타이밍 측정 및 PIPELINE_TIMING 로그 기록
"""

import logging
import os
import time
from typing import AsyncIterable, Optional

from livekit import rtc
from livekit.agents import Agent, ModelSettings, llm, stt

pipeline_timing_logger = logging.getLogger("PIPELINE_TIMING")
context_logger = logging.getLogger("CHAT_CONTEXT")
logger = logging.getLogger(__name__)

# Max chat context items to prevent token overflow (8192 token limit for most models)
# 대화는 Redis에 실시간 저장되고, 과거 맥락은 AutoRAG가 필요시 주입하므로
# LLM 컨텍스트에는 즉각적인 대화 흐름용으로 최근 6개만 유지
MAX_CONTEXT_ITEMS = int(os.getenv("MAX_CONTEXT_ITEMS", "6"))


class PipelineTimerMixin:
    """파이프라인 타이밍 측정을 위한 Mixin 클래스."""

    def __init__(self, *args, **kwargs):
        """pipeline_times 딕셔너리 초기화."""
        super().__init__(*args, **kwargs)
        self._pipeline_times: dict[str, float] = {}

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """Override STT node to measure STT timing."""
        events = Agent.default.stt_node(self, audio, model_settings)
        if events is None:
            pipeline_timing_logger.warning("STT node returned None - no STT configured")
            return None

        async def timed_stt_events():
            try:
                async for event in events:
                    if hasattr(event, "type"):
                        event_type = str(event.type)
                        if "final" in event_type.lower():
                            vad_end = self._pipeline_times.get("vad_end")
                            if vad_end:
                                stt_duration = time.time() - vad_end
                                self._pipeline_times["stt_duration"] = stt_duration
                                del self._pipeline_times["vad_end"]
                    yield event
            except Exception as e:
                pipeline_timing_logger.error(f"STT error: {e}")
                raise

        return timed_stt_events()

    def _estimate_tokens(self, text: str) -> int:
        """대략적인 토큰 수 추정 (한글 기준 ~2자당 1토큰, 영문 ~4자당 1토큰)."""
        if not text:
            return 0
        # 간단한 휴리스틱: 평균적으로 3자당 1토큰
        return len(text) // 3 + 1

    def _log_chat_context(self, chat_ctx: llm.ChatContext, phase: str) -> int:
        """ChatContext 상세 로깅. 총 추정 토큰 수 반환."""
        total_tokens = 0
        items_info = []

        for i, item in enumerate(chat_ctx.items):
            role = getattr(item, "role", "unknown")
            content = ""
            if hasattr(item, "content"):
                if isinstance(item.content, str):
                    content = item.content
                elif isinstance(item.content, list):
                    content = " ".join(str(c) for c in item.content)
                else:
                    content = str(item.content)

            tokens = self._estimate_tokens(content)
            total_tokens += tokens
            preview = content[:50].replace("\n", " ") + "..." if len(content) > 50 else content.replace("\n", " ")
            items_info.append(f"  [{i}] {role}: {tokens}tok | {preview}")

        context_logger.info(
            f"=== ChatContext ({phase}) ===\n"
            f"Items: {len(chat_ctx.items)} | EstTokens: ~{total_tokens}\n"
            + "\n".join(items_info)
        )
        return total_tokens

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list,
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Override LLM node to measure LLM timing."""
        # 트렁케이트 전 로깅
        items_before = len(chat_ctx.items)
        tokens_before = self._log_chat_context(chat_ctx, "BEFORE truncate")

        # Truncate context to prevent token overflow
        if items_before > MAX_CONTEXT_ITEMS:
            chat_ctx.truncate(max_items=MAX_CONTEXT_ITEMS)
            tokens_after = self._log_chat_context(chat_ctx, "AFTER truncate")
            context_logger.info(
                f"[TRUNCATE] {items_before} -> {len(chat_ctx.items)} items | "
                f"~{tokens_before} -> ~{tokens_after} tokens"
            )

        llm_start = time.time()
        self._pipeline_times["llm_start"] = llm_start
        first_chunk = True

        async for chunk in Agent.default.llm_node(
            self, chat_ctx, tools, model_settings
        ):
            if first_chunk:
                self._pipeline_times["llm_ttft"] = time.time() - llm_start
                first_chunk = False
            yield chunk

        self._pipeline_times["llm_duration"] = time.time() - llm_start

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Override TTS node to measure TTS timing and log pipeline summary."""
        tts_start = time.time()
        self._pipeline_times["tts_start"] = tts_start
        first_frame = True

        async for frame in Agent.default.tts_node(self, text, model_settings):
            if first_frame:
                self._pipeline_times["tts_ttfb"] = time.time() - tts_start
                first_frame = False
            yield frame

        self._pipeline_times["tts_duration"] = time.time() - tts_start
        self._log_pipeline_summary()

    def _log_pipeline_summary(self):
        """파이프라인 전체 소요 시간 로그."""
        vad = self._pipeline_times.get("vad_duration", 0)
        stt_time = self._pipeline_times.get("stt_duration", 0)
        rag = self._pipeline_times.get("rag_duration", 0)
        llm_time = self._pipeline_times.get("llm_duration", 0)
        tts_time = self._pipeline_times.get("tts_duration", 0)
        total = vad + stt_time + rag + llm_time + tts_time
        pipeline_timing_logger.info(
            f"TOTAL={total:.3f}s | VAD={vad:.3f}s STT={stt_time:.3f}s "
            f"RAG={rag:.3f}s LLM={llm_time:.3f}s TTS={tts_time:.3f}s"
        )
