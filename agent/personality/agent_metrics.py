"""
AgentMetrics - 파이프라인 타이밍 측정

역할: STT/LLM/TTS 노드의 타이밍 측정 및 PIPELINE_TIMING 로그 기록
"""

import logging
import os
import time
from typing import AsyncIterable, Optional

from livekit import rtc
from livekit.agents import Agent, ModelSettings, llm, stt

pipeline_timing_logger = logging.getLogger("PIPELINE_TIMING")

# MCP (Tool calling) toggle - disable for models that don't support tools
MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() == "true"


class AgentMetricsMixin:
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

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list,
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Override LLM node to measure LLM timing."""
        # Force empty tools if MCP disabled (for models without tool support)
        if not MCP_ENABLED:
            tools = []

        llm_start = time.time()
        self._pipeline_times["llm_start"] = llm_start
        first_chunk = True

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
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
