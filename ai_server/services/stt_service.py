"""
STT Service - Whisper Speech-to-Text (GPU).

faster-whisper 기반 비동기 음성 인식 서비스.
GPU 메모리 효율적인 처리와 동시 요청 제한 지원.

환경변수:
    WHISPER_MODEL: 모델명 (기본: large-v3-turbo)
    WHISPER_DEVICE: 장치 (기본: cuda)
    WHISPER_COMPUTE_TYPE: 연산 타입 (기본: float16)
    MAX_CONCURRENT_STT: 최대 동시 요청 (기본: 8)

사용:
    from services.stt_service import STTService

    stt = STTService()
    await stt.initialize()

    result = await stt.transcribe(audio_data, language="ko")
    print(result["text"])
"""

import asyncio
import io
import logging
import os
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================
# 설정
# ============================================
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
MAX_CONCURRENT_STT = int(os.getenv("MAX_CONCURRENT_STT", "8"))


# ============================================
# Whisper 모델 싱글톤
# ============================================
class WhisperSingleton:
    """Whisper 모델 싱글톤 - GPU 메모리 효율적인 관리."""

    _instance: Optional["WhisperSingleton"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self):
        """모델 로드 (lazy loading)."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
            logger.info(f"Device: {WHISPER_DEVICE}, Compute type: {WHISPER_COMPUTE_TYPE}")

            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                # 병렬 처리 최적화
                cpu_threads=4,
                num_workers=4,
            )
            logger.info("Whisper model loaded successfully")
        return self._model

    @property
    def model(self):
        """모델 인스턴스 반환."""
        if self._model is None:
            return self.load_model()
        return self._model

    def is_loaded(self) -> bool:
        """모델 로드 여부 확인."""
        return self._model is not None


# 전역 싱글톤 인스턴스
whisper_singleton = WhisperSingleton()


# ============================================
# 헬퍼 함수
# ============================================
def audio_bytes_to_array(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """오디오 바이트를 numpy 배열로 변환.

    Args:
        audio_bytes: 오디오 데이터 (WAV 또는 raw PCM)
        sample_rate: 샘플레이트 (PCM 변환 시 사용)

    Returns:
        float32 numpy 배열 (-1.0 ~ 1.0)

    Raises:
        ValueError: 오디오 파싱 실패 시
    """
    # WAV 파일 파싱 시도
    try:
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                return audio_array
    except Exception:
        pass

    # Raw PCM으로 처리 (16-bit signed, mono)
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_array
    except Exception as e:
        raise ValueError(f"Failed to parse audio data: {e}")


# ============================================
# STT 서비스
# ============================================
class STTService:
    """Whisper STT 서비스 - 비동기 음성 인식.

    동시 요청 제한과 GPU 메모리 관리를 위한 세마포어 사용.

    Attributes:
        model_name: 사용 중인 Whisper 모델명
        device: 사용 중인 장치 (cuda/cpu)
        compute_type: 연산 타입 (float16/float32)
        max_concurrent: 최대 동시 요청 수
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        max_concurrent: Optional[int] = None,
    ):
        """STT 서비스 초기화.

        Args:
            model: Whisper 모델명 (기본: 환경변수)
            device: 장치 (기본: 환경변수)
            compute_type: 연산 타입 (기본: 환경변수)
            max_concurrent: 최대 동시 요청 (기본: 환경변수)
        """
        self.model_name = model or WHISPER_MODEL
        self.device = device or WHISPER_DEVICE
        self.compute_type = compute_type or WHISPER_COMPUTE_TYPE
        self.max_concurrent = max_concurrent or MAX_CONCURRENT_STT

        self._semaphore: Optional[asyncio.Semaphore] = None
        self._initialized = False

    async def initialize(self) -> None:
        """서비스 초기화 - 모델 로드 및 세마포어 생성."""
        if self._initialized:
            return

        logger.info("Initializing STT Service...")
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # 모델 로드 (블로킹 방지를 위해 스레드풀에서 실행)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, whisper_singleton.load_model)

        self._initialized = True
        logger.info(f"STT Service initialized: max_concurrent={self.max_concurrent}")

    def is_initialized(self) -> bool:
        """서비스 초기화 여부 확인."""
        return self._initialized

    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> dict:
        """비동기 음성 인식 수행.

        Args:
            audio_data: 오디오 데이터 (WAV 또는 raw PCM bytes)
            language: 언어 코드 (예: "ko", "en"), None이면 자동 감지
            sample_rate: 샘플레이트 (raw PCM인 경우 사용)

        Returns:
            dict: {
                "text": 인식된 텍스트,
                "language": 감지된 언어,
                "language_probability": 언어 감지 확률,
                "duration": 오디오 길이(초),
                "segments": 세그먼트 목록
            }

        Raises:
            RuntimeError: 서비스가 초기화되지 않은 경우
            ValueError: 오디오 데이터 파싱 실패
        """
        if not self._initialized:
            raise RuntimeError("STT Service not initialized. Call initialize() first.")

        # 오디오 데이터 변환
        try:
            audio_array = audio_bytes_to_array(audio_data, sample_rate)
        except ValueError as e:
            raise ValueError(f"Audio parsing failed: {e}")

        # 리샘플링 (Whisper는 16kHz 필요)
        if sample_rate != 16000:
            import scipy.signal

            audio_array = scipy.signal.resample(
                audio_array,
                int(len(audio_array) * 16000 / sample_rate),
            )

        # 동시 요청 제한
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_array,
                language,
            )

    def _transcribe_sync(
        self,
        audio_array: np.ndarray,
        language: Optional[str],
    ) -> dict:
        """동기 음성 인식 (스레드풀에서 실행).

        할루시네이션 최소화 설정:
        - beam_size=1: 연구결과 할루시네이션 최소
        - condition_on_previous_text=False: 반복 루프 방지
        - no_speech_threshold=0.3: 낮춰서 노이즈 인식 방지
        - hallucination_silence_threshold=1.0: 긴 침묵 시 할루시네이션 스킵
        """
        model = whisper_singleton.model

        # 한국어 강제 지정 (자동 감지보다 정확도 높음)
        target_language = language if language else "ko"

        segments, info = model.transcribe(
            audio_array,
            language=target_language,
            beam_size=5,  # 정확도 향상 (1 -> 5)
            temperature=0.0,  # 결정론적 출력
            condition_on_previous_text=False,  # 반복 루프 방지
            no_speech_threshold=0.3,  # 낮춰서 노이즈를 음성으로 잘못 인식 방지
            hallucination_silence_threshold=1.0,  # 1초 이상 침묵 시 할루시네이션 스킵
            word_timestamps=True,  # hallucination_silence_threshold 작동 필요
            vad_filter=True,  # 침묵 구간 스킵
            vad_parameters={
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 100,
            },
            # 한국어 인식 개선 - 자주 쓰는 표현 포함
            initial_prompt="안녕하세요. 지금 몇 시야? 날씨 어때? 몇 도야? 밥 먹었어? 뭐 먹었어? 어디 갔어? 알려줘. 그래. 응. 네.",
        )

        # 세그먼트 수집
        segment_list = []
        full_text = ""

        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            })
            full_text += segment.text

        return {
            "text": full_text.strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segment_list,
        }

    async def transcribe_streaming(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
    ):
        """스트리밍 음성 인식 - 세그먼트 단위로 yield.

        Args:
            audio_data: 오디오 데이터
            language: 언어 코드

        Yields:
            dict: 세그먼트 정보 {"text", "start", "end", "is_final"}
        """
        if not self._initialized:
            raise RuntimeError("STT Service not initialized. Call initialize() first.")

        audio_array = audio_bytes_to_array(audio_data)

        async with self._semaphore:
            model = whisper_singleton.model
            target_language = language if language else "ko"
            segments, _ = model.transcribe(
                audio_array,
                language=target_language,
                beam_size=5,  # 정확도 향상
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.3,
                hallucination_silence_threshold=1.0,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 200,
                    "speech_pad_ms": 100,
                },
                initial_prompt="안녕하세요. 지금 몇 시야? 날씨 어때? 몇 도야? 밥 먹었어? 뭐 먹었어? 어디 갔어? 알려줘. 그래. 응. 네.",
            )

            for segment in segments:
                yield {
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end,
                    "is_final": True,
                }

    def get_status(self) -> dict:
        """서비스 상태 정보 반환."""
        return {
            "initialized": self._initialized,
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "max_concurrent": self.max_concurrent,
            "model_loaded": whisper_singleton.is_loaded(),
        }
