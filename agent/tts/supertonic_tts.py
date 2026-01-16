"""
Supertonic-2 ONNX TTS Plugin for LiveKit Agents.

다국어 지원 (한국어, 영어, 스페인어, 포르투갈어, 프랑스어)
Hugging Face: https://huggingface.co/Supertone/supertonic-2

설치:
    모델은 Docker 빌드 시 /app/models/supertonic-2에 다운로드됨

사용:
    from tts.supertonic_tts import SupertonicTTS

    # 기본 사용 (F2 여성 음성, 최고 품질)
    tts = SupertonicTTS()

    # 커스텀 설정
    tts = SupertonicTTS(voice="M1", total_steps=10, lang="en")

음성 스타일:
    - M1~M5: 남성 음성
    - F1~F5: 여성 음성

언어:
    - ko: 한국어
    - en: 영어
    - es: 스페인어
    - pt: 포르투갈어
    - fr: 프랑스어
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

from livekit.agents import tts, APIConnectOptions
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

# Voice style type
VoiceStyle = Literal["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
Language = Literal["ko", "en", "es", "pt", "fr"]

# 기본 모델 경로 (Docker 내부 - 볼륨 마운트에 영향받지 않는 위치)
DEFAULT_MODEL_PATH = "/opt/models/supertonic-2"


@dataclass
class SupertonicOptions:
    voice: VoiceStyle
    total_steps: int  # 품질: 2(낮음) ~ 15(높음)
    speed: float  # 속도: 0.7(느림) ~ 2.0(빠름)
    lang: Language  # 기본 언어


def detect_language(text: str) -> Language:
    """텍스트에서 언어 자동 감지 (한국어/영어)"""
    # 한글이 포함되어 있으면 한국어
    korean_pattern = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
    if korean_pattern.search(text):
        return "ko"
    return "en"


class SupertonicTTS(tts.TTS):
    """
    Supertonic-2 ONNX TTS - 다국어 온디바이스 TTS.

    Args:
        voice: 음성 스타일 (M1-M5: 남성, F1-F5: 여성), 기본 F2
        total_steps: 품질 단계 (2=저품질/빠름, 15=고품질/느림), 기본 15
        speed: 말하기 속도 (0.7~2.0), 기본 1.0
        lang: 기본 언어 ("ko", "en", "es", "pt", "fr"), 기본 "ko"
        model_path: ONNX 모델 경로, 기본 /app/models/supertonic-2
        auto_detect_lang: 텍스트에서 언어 자동 감지 여부, 기본 True
    """

    def __init__(
        self,
        *,
        voice: VoiceStyle = "F2",
        total_steps: int = 15,
        speed: float = 1.0,
        lang: Language = "ko",
        model_path: Optional[str] = None,
        auto_detect_lang: bool = True,
    ) -> None:
        # 모델 경로 설정
        self._model_path = model_path or os.environ.get(
            "SUPERTONIC_MODEL_PATH", DEFAULT_MODEL_PATH
        )
        self._auto_detect_lang = auto_detect_lang

        # Lazy import - helper 모듈 로드
        try:
            from .supertonic_helper import (
                load_text_to_speech,
                load_voice_style,
            )
        except ImportError as e:
            raise ImportError(
                "supertonic_helper not found. Ensure supertonic_helper.py is in the tts directory."
            ) from e

        # ONNX 모델 로드
        onnx_dir = os.path.join(self._model_path, "onnx")
        voice_styles_dir = os.path.join(self._model_path, "voice_styles")

        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_dir}. "
                "Run: git lfs install && git clone https://huggingface.co/Supertone/supertonic-2"
            )

        logger.info(f"Loading Supertonic-2 ONNX model from {onnx_dir}")
        self._engine = load_text_to_speech(onnx_dir)

        # 샘플레이트는 모델에서 가져옴
        sample_rate = self._engine.sample_rate

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = SupertonicOptions(
            voice=voice,
            total_steps=total_steps,
            speed=speed,
            lang=lang,
        )

        # 음성 스타일 로드
        voice_style_path = os.path.join(voice_styles_dir, f"{voice}.json")
        if not os.path.exists(voice_style_path):
            raise FileNotFoundError(
                f"Voice style not found: {voice_style_path}"
            )
        self._voice_style = load_voice_style([voice_style_path])

        logger.info(
            f"SupertonicTTS initialized: voice={voice}, "
            f"steps={total_steps}, speed={speed}, lang={lang}, "
            f"sample_rate={sample_rate}, auto_detect={auto_detect_lang}"
        )

    @property
    def model(self) -> str:
        return f"supertonic-2-{self._opts.voice}"

    @property
    def provider(self) -> str:
        return "Supertonic"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SupertonicChunkedStream":
        return SupertonicChunkedStream(
            tts=self,
            text=text,
            conn_options=conn_options,
        )


class SupertonicChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: SupertonicTTS,
        text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import numpy as np

        request_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()

        # 언어 감지
        if self._tts._auto_detect_lang:
            lang = detect_language(self._input_text)
        else:
            lang = self._tts._opts.lang

        try:
            # Supertonic ONNX는 동기 API이므로 스레드풀에서 실행
            wav, duration = await loop.run_in_executor(
                None,
                lambda: self._tts._engine(
                    text=self._input_text,
                    lang=lang,
                    style=self._tts._voice_style,
                    total_step=self._tts._opts.total_steps,
                    speed=self._tts._opts.speed,
                ),
            )

            # wav shape: (1, num_samples) → squeeze → (num_samples,)
            # float32 [-1, 1] → int16 PCM
            audio_float = wav.squeeze()
            audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            logger.debug(
                f"SupertonicTTS synthesized: {len(self._input_text)} chars → "
                f"{duration[0]:.2f}s audio (lang={lang})"
            )

            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._tts.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )
            output_emitter.push(audio_bytes)

        except Exception as e:
            logger.error(f"SupertonicTTS synthesis failed: {e}")
            raise
