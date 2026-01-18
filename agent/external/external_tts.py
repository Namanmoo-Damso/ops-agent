"""
External TTS implementation for LiveKit Agents.

HTTP-based TTS client that calls AI Server /v1/audio/speech.
Implements LiveKit TTS protocol for seamless integration.
"""

import logging
import uuid
from typing import Optional

import aiohttp
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger(__name__)


class ExternalTTSStream(tts.ChunkedStream):
    """
    Chunked stream for External TTS.

    Handles the HTTP request and emits audio chunks as they arrive.
    """

    def __init__(
        self,
        *,
        tts_instance: "ExternalTTS",
        input_text: str,
        conn_options: APIConnectOptions,
        voice: str,
    ):
        super().__init__(
            tts=tts_instance,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._voice = voice

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Execute TTS request and stream audio to emitter.
        """
        request_id = str(uuid.uuid4())
        tts_instance: ExternalTTS = self._tts  # type: ignore

        # Initialize emitter with audio format
        # AI Server returns raw PCM audio (16-bit, mono)
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=tts_instance.sample_rate,
            num_channels=tts_instance.num_channels,
            mime_type="audio/pcm",
        )

        try:
            session = await tts_instance._ensure_session()

            url = f"{tts_instance._base_url}/v1/audio/speech"

            # Prepare form data (AI Server expects Form, not JSON)
            form_data = aiohttp.FormData()
            form_data.add_field("text", self._input_text)
            form_data.add_field("voice", self._voice)
            form_data.add_field("output_format", "pcm")

            logger.debug(f"Calling TTS API: {url} with voice={self._voice}")

            async with session.post(url, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"TTS API error: {response.status} - {error_text}")
                    from livekit.agents._exceptions import APIError
                    raise APIError(f"TTS API error: {response.status} - {error_text}")

                # Stream audio chunks as they arrive
                async for chunk in response.content.iter_chunked(4096):
                    if chunk:
                        output_emitter.push(chunk)

            logger.debug(f"TTS completed for text: '{self._input_text[:50]}...'")

        except aiohttp.ClientError as e:
            logger.error(f"TTS HTTP client error: {e}")
            from livekit.agents._exceptions import APIError
            raise APIError(f"TTS HTTP error: {e}") from e
        except Exception as e:
            logger.exception(f"TTS unexpected error: {e}")
            from livekit.agents._exceptions import APIError
            raise APIError(f"TTS error: {e}") from e


class ExternalTTS(tts.TTS):
    """
    External TTS client that calls AI Server for text-to-speech.

    Implements LiveKit's TTS protocol using HTTP POST to /v1/audio/speech.
    Returns PCM audio at 24kHz sample rate.
    """

    def __init__(
        self,
        base_url: str,
        *,
        voice: str = "F2",
        sample_rate: int = 44100,
        timeout: float = 60.0,
        model: str = "external-tts",
    ):
        """
        Initialize External TTS client.

        Args:
            base_url: AI Server base URL (e.g., "http://ai-server:8000")
            voice: Voice ID to use (default: "F2" for Korean female)
            sample_rate: Audio sample rate (default: 24000 Hz)
            timeout: Request timeout in seconds (default: 60.0)
            model: Model name for identification
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # HTTP API returns full audio
            ),
            sample_rate=sample_rate,
            num_channels=1,  # Mono audio
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._timeout = timeout
        self._model_name = model
        self._http_session: Optional[aiohttp.ClientSession] = None

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "external-ai-server"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available and reuse for connection pooling."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        return self._http_session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ExternalTTSStream:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            ExternalTTSStream that yields audio chunks
        """
        return ExternalTTSStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
            voice=self._voice,
        )

    async def aclose(self) -> None:
        """Clean up HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
