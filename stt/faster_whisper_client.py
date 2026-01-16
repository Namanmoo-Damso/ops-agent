import io
import logging
import wave

import aiohttp
from livekit.agents import stt

logger = logging.getLogger(__name__)


class FasterWhisperSTT(stt.STT):
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        language: str = "ko",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_threshold: float = 0.5,
        initial_prompt: str = "",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._server_url = server_url.rstrip("/")
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._vad_threshold = vad_threshold
        self._initial_prompt = initial_prompt
        self._session: aiohttp.ClientSession | None = None

        logger.info(f"FasterWhisperSTT (v1.3.10) initialized: {self._server_url}")

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)

    def _buffer_to_wav(self, buffer) -> bytes:
        """
        다양한 타입의 오디오 버퍼/프레임을 WAV 바이너리로 변환합니다.
        """
        # 1. PCM 데이터 추출 (타입별 대응)
        if hasattr(buffer, "export"):
            # stt.AudioBuffer 형태
            pcm_data = buffer.export()
        elif hasattr(buffer, "data"):
            # rtc.AudioFrame 형태
            pcm_data = bytes(buffer.data)
        elif isinstance(buffer, list):
            # 프레임 리스트 형태
            pcm_data = b"".join(bytes(f.data) for f in buffer)
        else:
            raise ValueError(f"Unsupported audio buffer type: {type(buffer)}")

        # 2. WAV 파일 생성
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
        language: str | None = None,
        **kwargs,
    ) -> stt.SpeechEvent:
        await self._ensure_session()

        try:
            # WAV 변환
            audio_data = self._buffer_to_wav(buffer)

            # 디버그 로그
            logger.info(f"--- [STT] Sending audio: {len(audio_data)} bytes ---")

            lang = language or self._language
            url = f"{self._server_url}/v1/audio/transcriptions"

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

            async with self._session.post(url, data=form) as resp:
                resp.raise_for_status()
                result = await resp.json()

            text = result.get("text", "").strip()
            if not text:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[]
                )

            logger.info(f"--- [STT] Result: {text} ---")

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text=text, language=lang, confidence=1.0)],
            )
        except Exception as e:
            logger.error(f"--- [STT] Error: {e} ---")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[]
            )

    def stream(self, **kwargs):
        raise NotImplementedError("FasterWhisperSTT does not support streaming")

    async def aclose(self):
        if self._session and not self._session.closed:
            await self._session.close()
