"""
External STT implementation for LiveKit Agents.

HTTP-based STT client that calls AI Server /v1/audio/transcriptions.
Non-streaming mode: AgentSession handles VAD, then calls recognize().
"""

import io
import logging
import wave
from typing import Optional

import aiohttp
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

logger = logging.getLogger(__name__)


class ExternalSTT(stt.STT):
    """
    External STT client that calls AI Server for speech-to-text.

    Non-streaming mode: AgentSession handles VAD and calls _recognize_impl()
    with the recorded audio buffer.
    """

    def __init__(
        self,
        base_url: str,
        *,
        language: str = "ko",
        timeout: float = 30.0,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
        initial_prompt: str = "",
    ):
        """
        Initialize External STT client.

        Args:
            base_url: AI Server base URL (e.g., "http://localhost:8001")
            language: Default language code (default: "ko" for Korean)
            timeout: Request timeout in seconds (default: 30.0)
            beam_size: Whisper beam size (default: 5)
            vad_filter: Enable VAD filter (default: True)
            vad_threshold: VAD threshold (default: 0.5)
            initial_prompt: Initial prompt for Whisper (default: "")
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,  # Non-streaming: AgentSession handles VAD
                interim_results=False,
            )
        )
        self._base_url = base_url.rstrip("/")
        self._language = language
        self._timeout = timeout
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._vad_threshold = vad_threshold
        self._initial_prompt = initial_prompt
        self._http_session: Optional[aiohttp.ClientSession] = None

        logger.info(f"ExternalSTT initialized: {self._base_url}")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available and reuse for connection pooling."""
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout, connect=5)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    def _buffer_to_wav(self, buffer) -> bytes:
        """
        Convert various audio buffer/frame types to WAV binary.
        """
        # 1. Extract PCM data (handle different types)
        if hasattr(buffer, "export"):
            # stt.AudioBuffer type
            pcm_data = buffer.export()
        elif hasattr(buffer, "data"):
            # rtc.AudioFrame type
            pcm_data = bytes(buffer.data)
        elif isinstance(buffer, list):
            # List of frames
            pcm_data = b"".join(bytes(f.data) for f in buffer)
        else:
            raise ValueError(f"Unsupported audio buffer type: {type(buffer)}")

        # 2. Create WAV file (fixed 16kHz mono 16-bit)
        sample_rate = 16000
        sample_width = 2
        channels = 1

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        return wav_buffer.getvalue()

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """
        Non-streaming recognition - sends audio buffer to AI Server.

        Called by AgentSession after VAD detects end of speech.

        Args:
            buffer: AudioBuffer containing recorded speech
            language: Language code override
            conn_options: API connection options

        Returns:
            SpeechEvent with transcription result
        """
        await self._ensure_session()

        lang = language if language is not NOT_GIVEN else self._language

        try:
            # Convert buffer to WAV
            audio_data = self._buffer_to_wav(buffer)

            logger.info(f"[ExternalSTT] Sending {len(audio_data)} bytes to API")

            url = f"{self._base_url}/v1/audio/transcriptions"

            # Prepare form data (same as FasterWhisperSTT)
            form = aiohttp.FormData()
            form.add_field(
                "file", audio_data, filename="audio.wav", content_type="audio/wav"
            )
            form.add_field("language", lang)
            form.add_field("beam_size", str(self._beam_size))
            form.add_field("vad_filter", str(self._vad_filter).lower())
            form.add_field("initial_prompt", str(self._initial_prompt))

            if self._vad_filter:
                form.add_field("vad_threshold", str(self._vad_threshold))

            async with self._http_session.post(url, data=form) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[ExternalSTT] API error: {resp.status} - {error_text}")
                    return stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[],
                    )

                result = await resp.json()

            text = result.get("text", "").strip()
            if not text:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[],
                )

            logger.info(f"[ExternalSTT] Transcription: '{text}'")

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=text,
                        language=lang,
                        confidence=1.0,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"[ExternalSTT] Recognition error: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[],
            )

    async def aclose(self) -> None:
        """Clean up HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
