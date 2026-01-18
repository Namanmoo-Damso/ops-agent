"""
TTS Service - Supertonic-2 Text-to-Speech (GPU).

Supertonic-2 ONNX 기반 다국어 TTS 서비스.
GPU 지원으로 빠른 음성 합성.

환경변수:
    TTS_DEVICE: 장치 (기본: gpu, 옵션: cpu)
    TTS_VOICE: 음성 스타일 (기본: F2)
    TTS_SPEED: 말하기 속도 (기본: 1.0)
    TTS_TOTAL_STEPS: 품질 단계 (기본: 15)
    SUPERTONIC_MODEL_PATH: 모델 경로 (기본: /opt/models/supertonic-2)

사용:
    from services.tts_service import TTSService

    tts = TTSService()
    await tts.initialize()

    audio_bytes = await tts.synthesize("안녕하세요", voice="F2", speed=1.0)
"""

import asyncio
import logging
import os
import re
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================
# 설정
# ============================================
TTS_DEVICE = os.getenv("TTS_DEVICE", "gpu")
TTS_VOICE = os.getenv("TTS_VOICE", "F2")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_TOTAL_STEPS = int(os.getenv("TTS_TOTAL_STEPS", "15"))
SUPERTONIC_MODEL_PATH = os.getenv("SUPERTONIC_MODEL_PATH", "/opt/models/supertonic-2")

# Voice style type
VoiceStyle = Literal["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
Language = Literal["ko", "en", "es", "pt", "fr"]


def detect_language(text: str) -> Language:
    """텍스트에서 언어 자동 감지 (한국어/영어).

    Args:
        text: 입력 텍스트

    Returns:
        감지된 언어 코드 ("ko" 또는 "en")
    """
    korean_pattern = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
    if korean_pattern.search(text):
        return "ko"
    return "en"


# ============================================
# TTS 서비스
# ============================================
class TTSService:
    """Supertonic-2 TTS 서비스 - 비동기 음성 합성.

    다국어 지원 (한국어, 영어, 스페인어, 포르투갈어, 프랑스어).

    Attributes:
        model_path: ONNX 모델 경로
        use_gpu: GPU 사용 여부
        default_voice: 기본 음성 스타일
        default_speed: 기본 말하기 속도
        sample_rate: 오디오 샘플레이트
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        default_voice: Optional[VoiceStyle] = None,
        default_speed: Optional[float] = None,
        total_steps: Optional[int] = None,
    ):
        """TTS 서비스 초기화.

        Args:
            model_path: ONNX 모델 경로 (기본: 환경변수)
            use_gpu: GPU 사용 여부 (기본: 환경변수)
            default_voice: 기본 음성 스타일 (기본: 환경변수)
            default_speed: 기본 말하기 속도 (기본: 환경변수)
            total_steps: 품질 단계 (기본: 환경변수)
        """
        self.model_path = model_path or SUPERTONIC_MODEL_PATH
        self.use_gpu = use_gpu if use_gpu is not None else (TTS_DEVICE.lower() in ("gpu", "cuda"))
        self.default_voice = default_voice or TTS_VOICE
        self.default_speed = default_speed or TTS_SPEED
        self.total_steps = total_steps or TTS_TOTAL_STEPS

        self._engine = None
        self._voice_styles: dict = {}
        self._initialized = False
        self.sample_rate = 22050  # 기본값, 초기화 시 업데이트

    async def initialize(self) -> None:
        """서비스 초기화 - 모델 로드."""
        if self._initialized:
            return

        logger.info("Initializing TTS Service...")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Use GPU: {self.use_gpu}")

        # 모델 경로 검증
        onnx_dir = os.path.join(self.model_path, "onnx")
        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_dir}. "
                "Ensure the model is downloaded to the correct path."
            )

        # 모델 로드 (블로킹 방지를 위해 스레드풀에서 실행)
        loop = asyncio.get_event_loop()
        self._engine = await loop.run_in_executor(
            None,
            self._load_engine,
            onnx_dir,
        )

        # 샘플레이트 업데이트
        self.sample_rate = self._engine.sample_rate

        # 기본 음성 스타일 로드
        await self._load_voice_style(self.default_voice)

        self._initialized = True
        logger.info(
            f"TTS Service initialized: voice={self.default_voice}, "
            f"speed={self.default_speed}, steps={self.total_steps}, "
            f"sample_rate={self.sample_rate}, gpu={self.use_gpu}"
        )

    def _load_engine(self, onnx_dir: str):
        """ONNX 엔진 로드 (동기)."""
        from .supertonic_helper import load_text_to_speech

        return load_text_to_speech(onnx_dir, use_gpu=self.use_gpu)

    async def _load_voice_style(self, voice: VoiceStyle) -> None:
        """음성 스타일 로드 (캐싱)."""
        if voice in self._voice_styles:
            return

        voice_styles_dir = os.path.join(self.model_path, "voice_styles")
        voice_style_path = os.path.join(voice_styles_dir, f"{voice}.json")

        if not os.path.exists(voice_style_path):
            raise FileNotFoundError(f"Voice style not found: {voice_style_path}")

        loop = asyncio.get_event_loop()
        from .supertonic_helper import load_voice_style

        self._voice_styles[voice] = await loop.run_in_executor(
            None,
            load_voice_style,
            [voice_style_path],
        )
        logger.debug(f"Loaded voice style: {voice}")

    def is_initialized(self) -> bool:
        """서비스 초기화 여부 확인."""
        return self._initialized

    async def synthesize(
        self,
        text: str,
        voice: Optional[VoiceStyle] = None,
        speed: Optional[float] = None,
        language: Optional[Language] = None,
        total_steps: Optional[int] = None,
        output_format: Literal["pcm", "wav"] = "pcm",
    ) -> bytes:
        """비동기 음성 합성.

        Args:
            text: 합성할 텍스트
            voice: 음성 스타일 (M1-M5: 남성, F1-F5: 여성)
            speed: 말하기 속도 (0.7~2.0)
            language: 언어 코드 ("ko", "en" 등), None이면 자동 감지
            total_steps: 품질 단계 (2~15)
            output_format: 출력 형식 ("pcm" 또는 "wav")

        Returns:
            bytes: 오디오 데이터 (int16 PCM 또는 WAV)

        Raises:
            RuntimeError: 서비스가 초기화되지 않은 경우
            ValueError: 잘못된 파라미터
        """
        if not self._initialized:
            raise RuntimeError("TTS Service not initialized. Call initialize() first.")

        # 기본값 설정
        voice = voice or self.default_voice
        speed = speed or self.default_speed
        total_steps = total_steps or self.total_steps

        # 언어 감지
        if language is None:
            language = detect_language(text)

        # 음성 스타일 로드 (필요시)
        await self._load_voice_style(voice)
        style = self._voice_styles[voice]

        # 음성 합성 (스레드풀에서 실행)
        loop = asyncio.get_event_loop()
        wav, duration = await loop.run_in_executor(
            None,
            lambda: self._engine(
                text=text,
                lang=language,
                style=style,
                total_step=total_steps,
                speed=speed,
            ),
        )

        logger.debug(
            f"TTS synthesized: {len(text)} chars -> {duration[0]:.2f}s audio "
            f"(voice={voice}, lang={language}, speed={speed})"
        )

        # 오디오 변환
        # wav shape: (1, num_samples) -> squeeze -> (num_samples,)
        # float32 [-1, 1] -> int16 PCM
        audio_float = wav.squeeze()
        audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)

        if output_format == "wav":
            return self._to_wav(audio_int16)
        return audio_int16.tobytes()

    def _to_wav(self, audio_int16: np.ndarray) -> bytes:
        """int16 PCM을 WAV 형식으로 변환."""
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    async def synthesize_streaming(
        self,
        text: str,
        voice: Optional[VoiceStyle] = None,
        speed: Optional[float] = None,
        language: Optional[Language] = None,
        chunk_size: int = 4096,
    ):
        """스트리밍 음성 합성 - 청크 단위로 yield.

        Args:
            text: 합성할 텍스트
            voice: 음성 스타일
            speed: 말하기 속도
            language: 언어 코드
            chunk_size: 청크 크기 (bytes)

        Yields:
            bytes: 오디오 청크 (int16 PCM)
        """
        # 전체 오디오 생성 후 청크로 분할
        # (Supertonic은 스트리밍 생성을 지원하지 않음)
        audio_bytes = await self.synthesize(text, voice, speed, language)

        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    def get_available_voices(self) -> list[str]:
        """사용 가능한 음성 스타일 목록 반환."""
        return ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

    def get_available_languages(self) -> list[str]:
        """사용 가능한 언어 목록 반환."""
        return ["ko", "en", "es", "pt", "fr"]

    def get_status(self) -> dict:
        """서비스 상태 정보 반환."""
        return {
            "initialized": self._initialized,
            "model_path": self.model_path,
            "use_gpu": self.use_gpu,
            "default_voice": self.default_voice,
            "default_speed": self.default_speed,
            "total_steps": self.total_steps,
            "sample_rate": self.sample_rate,
            "loaded_voices": list(self._voice_styles.keys()),
        }
