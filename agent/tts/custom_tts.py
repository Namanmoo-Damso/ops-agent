"""
Custom TTS Plugin Template for LiveKit Agents.

이 템플릿을 수정하여 자체 TTS 모델을 연결하세요.
ONNX, HTTP API, 로컬 모델 등 어떤 방식이든 가능합니다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Awaitable

from livekit.agents import tts, APIConnectOptions
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)


@dataclass
class CustomTTSOptions:
    """TTS 옵션 설정"""

    voice: str
    sample_rate: int
    # 필요한 옵션 추가 (model_path, api_url 등)


class CustomTTS(tts.TTS):
    """
    커스텀 TTS 구현체.

    사용법:
        tts = CustomTTS(voice="korean-female", sample_rate=22050)

        # AgentSession에서 사용
        session = AgentSession(
            tts=tts,
            ...
        )
    """

    def __init__(
        self,
        *,
        voice: str = "default",
        sample_rate: int = 22050,
        # 추가 파라미터 (model_path, api_url 등)
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # 비스트리밍 (한 번에 전체 오디오 생성)
            ),
            sample_rate=sample_rate,
            num_channels=1,  # mono
        )

        self._opts = CustomTTSOptions(
            voice=voice,
            sample_rate=sample_rate,
        )

        # ============================================================
        # TODO: 여기에 모델 초기화 코드 추가
        # ============================================================
        # 예시 1: ONNX 모델 로드
        # import onnxruntime as ort
        # self._session = ort.InferenceSession(model_path)

        # 예시 2: HTTP API 클라이언트
        # self._api_url = api_url

        # 예시 3: 로컬 TTS 라이브러리
        # from TTS.api import TTS as CoquiTTS
        # self._tts_model = CoquiTTS(model_name="...")

        logger.info(f"CustomTTS initialized: voice={voice}, sample_rate={sample_rate}")

    @property
    def model(self) -> str:
        return "custom-tts"

    @property
    def provider(self) -> str:
        return "Custom"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "CustomChunkedStream":
        return CustomChunkedStream(
            tts=self,
            text=text,
            conn_options=conn_options,
        )


class CustomChunkedStream(tts.ChunkedStream):
    """TTS 합성 스트림 - 실제 오디오 생성이 여기서 수행됨"""

    def __init__(
        self,
        *,
        tts: CustomTTS,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        실제 TTS 합성을 수행하는 메서드.

        이 메서드를 수정하여 자체 TTS 모델을 연결하세요.
        """
        request_id = str(uuid.uuid4())

        # ============================================================
        # TODO: 여기에 TTS 합성 로직 구현
        # ============================================================

        # 예시 1: ONNX 모델 사용
        # audio_bytes = await self._synthesize_onnx(self._input_text)

        # 예시 2: HTTP API 호출
        # audio_bytes = await self._synthesize_api(self._input_text)

        # 예시 3: 로컬 TTS 라이브러리
        # audio_bytes = await self._synthesize_local(self._input_text)

        # 현재는 플레이스홀더 (무음)
        # 실제 구현 시 이 부분을 교체하세요
        logger.warning(f"CustomTTS: 플레이스홀더 - 실제 TTS 구현 필요: {self._input_text[:50]}")
        duration_sec = 0.5  # 0.5초 무음
        samples = int(self._tts._opts.sample_rate * duration_sec)
        audio_bytes = b"\x00\x00" * samples  # int16 silence

        # ============================================================
        # AudioEmitter로 오디오 전송 (이 부분은 수정 불필요)
        # ============================================================
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",  # Raw PCM int16
        )
        output_emitter.push(audio_bytes)

    # ================================================================
    # 아래는 구현 예시 메서드들 - 필요한 것만 사용
    # ================================================================

    async def _synthesize_onnx(self, text: str) -> bytes:
        """
        ONNX 모델로 TTS 합성 예시.

        import numpy as np
        import onnxruntime as ort

        # 텍스트 → 시퀀스 변환 (모델에 따라 다름)
        input_seq = self._text_to_sequence(text)

        # ONNX 추론
        outputs = self._tts._session.run(None, {"input": input_seq})
        audio = outputs[0]  # float32 waveform [-1, 1]

        # int16 PCM으로 변환
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
        """
        raise NotImplementedError("ONNX TTS 구현 필요")

    async def _synthesize_api(self, text: str) -> bytes:
        """
        HTTP API로 TTS 합성 예시.

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._tts._api_url,
                json={"text": text, "voice": self._tts._opts.voice},
                timeout=10.0,
            )
            return response.content  # PCM bytes
        """
        raise NotImplementedError("API TTS 구현 필요")

    async def _synthesize_local(self, text: str) -> bytes:
        """
        로컬 TTS 라이브러리 (Coqui TTS 등) 예시.

        import asyncio
        import numpy as np

        # 블로킹 호출을 스레드풀에서 실행
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: self._tts._tts_model.tts(text)
        )

        # float → int16 변환
        audio_int16 = (np.array(audio) * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
        """
        raise NotImplementedError("로컬 TTS 구현 필요")
