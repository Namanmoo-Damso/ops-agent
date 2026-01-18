"""
Faster-Whisper STT implementation for LiveKit Agents.

Uses large-v3-turbo model for fast, accurate Korean speech recognition.
"""

import logging
from typing import Union

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.stt import (
    STT,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

logger = logging.getLogger(__name__)

# Type alias for AudioBuffer
AudioBuffer = Union[list[rtc.AudioFrame], rtc.AudioFrame]


class FasterWhisperSTT(STT):
    """
    Faster-Whisper based STT for local GPU inference.

    Optimized for Korean elderly speech recognition using large-v3-turbo model.
    """

    def __init__(
        self,
        *,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ko",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: dict | None = None,
    ):
        """
        Initialize Faster-Whisper STT.

        Args:
            model_size: Whisper model size (default: large-v3-turbo)
            device: Device to run inference on (cuda/cpu)
            compute_type: Computation type (float16/int8/float32)
            language: Target language code (default: ko for Korean)
            beam_size: Beam size for decoding (default: 5)
            vad_filter: Filter out non-speech (breathing, noise) to prevent hallucination
            vad_parameters: Custom VAD parameters for filtering

        Note:
            Two VADs serve different purposes:
            - StreamAdapter VAD: Detects when user stops speaking (turn detection)
            - Whisper vad_filter: Filters noise/breathing in audio (hallucination prevention)
        """
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,  # Whisper doesn't support streaming
                interim_results=False,
            )
        )

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._vad_parameters = vad_parameters or {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }

        self._model: WhisperModel | None = None

    def _ensure_model_loaded(self) -> WhisperModel:
        """Lazy load the model on first use."""
        if self._model is None:
            logger.info(
                f"Loading Faster-Whisper model: {self._model_size} "
                f"(device={self._device}, compute_type={self._compute_type})"
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Faster-Whisper model loaded successfully")
        return self._model

    @property
    def model(self) -> str:
        return self._model_size

    @property
    def provider(self) -> str:
        return "faster-whisper"

    def _audio_buffer_to_numpy(self, buffer: AudioBuffer) -> tuple[np.ndarray, int]:
        """
        Convert AudioBuffer to numpy array for Faster-Whisper.

        Args:
            buffer: LiveKit AudioBuffer (list of AudioFrames or single AudioFrame)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Handle single frame or list of frames
        if isinstance(buffer, rtc.AudioFrame):
            frames = [buffer]
        else:
            frames = buffer

        if not frames:
            return np.array([], dtype=np.float32), 16000

        # Get sample rate from first frame
        sample_rate = frames[0].sample_rate

        # Concatenate all frames
        audio_data = []
        for frame in frames:
            # Get raw bytes and convert to numpy
            # AudioFrame data is int16 PCM
            frame_array = np.frombuffer(frame.data, dtype=np.int16)
            audio_data.append(frame_array)

        # Concatenate all frames
        audio_array = np.concatenate(audio_data)

        # Convert to float32 normalized to [-1, 1] for Whisper
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Whisper expects mono audio
        if frames[0].num_channels > 1:
            # Average channels for stereo to mono conversion
            audio_float = audio_float.reshape(-1, frames[0].num_channels).mean(axis=1)

        return audio_float, sample_rate

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        """
        Recognize speech from audio buffer using Faster-Whisper.

        Args:
            buffer: Audio data to transcribe
            language: Language code (overrides default if provided)
            conn_options: Connection options (unused for local inference)

        Returns:
            SpeechEvent with transcription result
        """
        import asyncio

        model = self._ensure_model_loaded()

        # Convert audio buffer to numpy array
        audio_array, sample_rate = self._audio_buffer_to_numpy(buffer)

        if len(audio_array) == 0:
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="", language=self._language)],
            )

        # Use provided language or default
        lang = language if language is not NOT_GIVEN else self._language

        # Run transcription in thread pool to avoid blocking
        def _transcribe():
            segments, info = model.transcribe(
                audio_array,
                language=lang,
                beam_size=self._beam_size,
                vad_filter=self._vad_filter,  # Filter noise/breathing to prevent hallucination
                vad_parameters=self._vad_parameters,
            )
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            return " ".join(text_parts), info.language

        loop = asyncio.get_event_loop()
        text, detected_language = await loop.run_in_executor(None, _transcribe)

        logger.debug(f"Transcribed: '{text}' (language: {detected_language})")

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    text=text,
                    language=detected_language or self._language,
                )
            ],
        )
