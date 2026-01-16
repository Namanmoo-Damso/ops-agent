"""
Custom TTS implementation for connecting to a self-hosted TTS server.

This module allows you to use your own TTS model running on a GPU instance
(e.g., EC2 with GPU) with LiveKit Agents.

Features:
- HTTP streaming for standard use
- WebSocket streaming for lowest latency
- Optional fallback to backup TTS

Usage:
    from tts import CustomTTS

    # Basic usage
    tts = CustomTTS(
        base_url="http://your-gpu-instance:8000",
        voice="default",
        sample_rate=24000,
    )

    # With WebSocket for lower latency
    tts = CustomTTS(
        base_url="http://your-gpu-instance:8000",
        use_websocket=True,
    )

    # With fallback to AWS Polly
    from tts import create_tts_with_fallback
    from livekit.plugins import aws

    tts = create_tts_with_fallback(
        CustomTTS(base_url="http://gpu:8000"),
        aws.TTS(voice="Seoyeon"),
    )

    session = AgentSession(
        tts=tts,
        # ... other config
    )
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

import aiohttp
from livekit.agents.tts import (
    TTS,
    ChunkedStream,
    SynthesizeStream,
    TTSCapabilities,
    FallbackAdapter,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger(__name__)


@dataclass
class CustomTTSOptions:
    """Configuration options for CustomTTS."""

    base_url: str
    voice: str = "default"
    speed: float = 1.0
    sample_rate: int = 24000
    num_channels: int = 1
    # API endpoints (customize if your server uses different paths)
    synthesize_endpoint: str = "/synthesize"
    stream_endpoint: str = "/stream"
    ws_endpoint: str = "/ws/synthesize"
    # Use WebSocket for lower latency streaming
    use_websocket: bool = False


class CustomTTS(TTS):
    """
    Custom TTS that connects to a self-hosted TTS API server.

    Your TTS server should expose:
    - POST /synthesize: One-shot synthesis (text -> full audio)
    - POST /stream: Streaming synthesis (text -> chunked audio)

    Both endpoints should accept JSON: {"text": "...", "voice": "..."}
    and return raw PCM audio (16-bit, mono).
    """

    def __init__(
        self,
        *,
        base_url: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        num_channels: int = 1,
        synthesize_endpoint: str = "/synthesize",
        stream_endpoint: str = "/stream",
        ws_endpoint: str = "/ws/synthesize",
        use_websocket: bool = False,
    ):
        """
        Initialize CustomTTS.

        Args:
            base_url: Base URL of your TTS server (e.g., "http://gpu-instance:8000")
            voice: Voice ID to use (passed to your server)
            speed: Speech speed (1.0 = normal, >1.0 = faster)
            sample_rate: Audio sample rate in Hz (default: 24000)
            num_channels: Number of audio channels (default: 1 for mono)
            synthesize_endpoint: API endpoint for one-shot synthesis
            stream_endpoint: API endpoint for streaming synthesis
            ws_endpoint: WebSocket endpoint for low-latency streaming
            use_websocket: Use WebSocket instead of HTTP for streaming (lower latency)
        """
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),  # Disabled - API changed in 1.3.10
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._opts = CustomTTSOptions(
            base_url=base_url.rstrip("/"),
            voice=voice,
            speed=speed,
            sample_rate=sample_rate,
            num_channels=num_channels,
            synthesize_endpoint=synthesize_endpoint,
            stream_endpoint=stream_endpoint,
            ws_endpoint=ws_endpoint,
            use_websocket=use_websocket,
        )

    @property
    def model(self) -> str:
        return "custom-tts"

    @property
    def provider(self) -> str:
        return "custom"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "CustomChunkedStream":
        """
        Synthesize text to audio (one-shot, non-streaming).

        Args:
            text: Text to synthesize
            conn_options: Connection options

        Returns:
            ChunkedStream that yields audio data
        """
        return CustomChunkedStream(
            tts=self,
            input_text=text,
            opts=self._opts,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SynthesizeStream":
        """
        Create a streaming synthesis session.

        Use push_text() to send text tokens, and iterate to receive audio.

        Args:
            conn_options: Connection options

        Returns:
            SynthesizeStream for streaming synthesis
        """
        if self._opts.use_websocket:
            return CustomWebSocketStream(
                tts=self,
                opts=self._opts,
                conn_options=conn_options,
            )
        return CustomSynthesizeStream(
            tts=self,
            opts=self._opts,
            conn_options=conn_options,
        )


class CustomChunkedStream(ChunkedStream):
    """
    ChunkedStream implementation for one-shot (non-streaming) synthesis.

    Sends full text to the TTS server and receives audio in chunks.
    """

    def __init__(
        self,
        *,
        tts: CustomTTS,
        input_text: str,
        opts: CustomTTSOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter) -> None:
        """
        Execute the synthesis request.

        This is called by the framework when audio is needed.
        """
        request_id = str(uuid.uuid4())
        text_len = len(self._input_text)
        text_preview = self._input_text[:50] + "..." if text_len > 50 else self._input_text

        logger.info(f"[CustomTTS] Synthesizing ({text_len} chars): {text_preview}")
        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self._opts.base_url}{self._opts.synthesize_endpoint}"

                async with session.post(
                    url,
                    json={
                        "text": self._input_text,
                        "voice": self._opts.voice,
                        "speed": self._opts.speed,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()

                    # Track time to first byte
                    first_byte_time = time.time()
                    ttfb = (first_byte_time - start_time) * 1000

                    # Initialize the emitter with audio format info
                    output_emitter.initialize(
                        request_id=request_id,
                        sample_rate=self._opts.sample_rate,
                        num_channels=self._opts.num_channels,
                        mime_type="audio/pcm",
                    )

                    # Stream audio chunks from response
                    total_bytes = 0
                    async for chunk in resp.content.iter_chunked(4096):
                        total_bytes += len(chunk)
                        output_emitter.push(chunk)

                    # Signal end of audio
                    output_emitter.flush()

                    # Calculate timing stats
                    total_time = (time.time() - start_time) * 1000
                    # Audio duration: bytes / (sample_rate * 2 bytes per sample * 1 channel)
                    audio_duration = total_bytes / (self._opts.sample_rate * 2) * 1000

                    logger.info(
                        f"[CustomTTS] Done: {text_len} chars, "
                        f"TTFB={ttfb:.0f}ms, total={total_time:.0f}ms, "
                        f"audio={audio_duration:.0f}ms ({total_bytes} bytes)"
                    )

        except aiohttp.ClientError as e:
            logger.error(f"[CustomTTS] HTTP error during synthesis: {e}")
            raise
        except Exception as e:
            logger.error(f"[CustomTTS] Synthesis failed: {e}")
            raise


class CustomSynthesizeStream(SynthesizeStream):
    """
    SynthesizeStream implementation for streaming synthesis.

    Receives text tokens via push_text() and streams audio back.
    This is used for real-time voice agents where latency matters.
    """

    def __init__(
        self,
        *,
        tts: CustomTTS,
        opts: CustomTTSOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter) -> None:
        """
        Main streaming synthesis loop.

        Receives text events from self._input and sends audio to output_emitter.
        """
        request_id = str(uuid.uuid4())

        logger.debug(f"[CustomTTS] Starting streaming synthesis: {request_id}")

        # Initialize output with audio format
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            mime_type="audio/pcm",
        )

        try:
            async with aiohttp.ClientSession() as session:
                # Process incoming text tokens
                async for event in self._input:
                    if event.token:
                        # Send text token to TTS server for synthesis
                        await self._synthesize_token(
                            session=session,
                            text=event.token,
                            output_emitter=output_emitter,
                        )

                    if event.is_flush:
                        # Segment boundary - flush the output
                        output_emitter.flush()

        except aiohttp.ClientError as e:
            logger.error(f"[CustomTTS] HTTP error during streaming: {e}")
            raise
        except Exception as e:
            logger.error(f"[CustomTTS] Streaming synthesis failed: {e}")
            raise
        finally:
            # Ensure final flush
            output_emitter.flush()
            logger.debug(f"[CustomTTS] Streaming synthesis complete: {request_id}")

    async def _synthesize_token(
        self,
        session: aiohttp.ClientSession,
        text: str,
        output_emitter,
    ) -> None:
        """
        Synthesize a single text token and stream the audio.

        Args:
            session: aiohttp session
            text: Text token to synthesize
            output_emitter: AudioEmitter to push audio data to
        """
        url = f"{self._opts.base_url}{self._opts.stream_endpoint}"

        try:
            async with session.post(
                url,
                json={
                    "text": text,
                    "voice": self._opts.voice,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()

                # Stream audio chunks
                async for chunk in resp.content.iter_chunked(4096):
                    output_emitter.push(chunk)

        except aiohttp.ClientError as e:
            logger.warning(f"[CustomTTS] Failed to synthesize token: {e}")
            # Don't raise - try to continue with remaining tokens


class CustomWebSocketStream(SynthesizeStream):
    """
    WebSocket-based streaming for lowest latency.

    Maintains a persistent WebSocket connection to the TTS server,
    sending text and receiving audio with minimal overhead.
    """

    def __init__(
        self,
        *,
        tts: CustomTTS,
        opts: CustomTTSOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter) -> None:
        """
        Main WebSocket streaming loop.

        Opens a single WebSocket connection and streams text/audio bidirectionally.
        """
        request_id = str(uuid.uuid4())

        logger.debug(f"[CustomTTS] Starting WebSocket streaming: {request_id}")

        # Initialize output with audio format
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=self._opts.num_channels,
            mime_type="audio/pcm",
        )

        # Convert http:// to ws:// for WebSocket
        ws_base = self._opts.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{ws_base}{self._opts.ws_endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    # Create tasks for sending and receiving
                    send_task = asyncio.create_task(
                        self._send_text(ws), name="ws_send"
                    )
                    recv_task = asyncio.create_task(
                        self._receive_audio(ws, output_emitter), name="ws_recv"
                    )

                    # Wait for send to complete, then signal end
                    await send_task

                    # Send end signal to server
                    await ws.send_str("")

                    # Wait for all audio to be received
                    await recv_task

        except aiohttp.ClientError as e:
            logger.error(f"[CustomTTS] WebSocket error: {e}")
            raise
        except Exception as e:
            logger.error(f"[CustomTTS] WebSocket streaming failed: {e}")
            raise
        finally:
            output_emitter.flush()
            logger.debug(f"[CustomTTS] WebSocket streaming complete: {request_id}")

    async def _send_text(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send text tokens to the WebSocket."""
        async for event in self._input:
            if event.token:
                # Send text with voice info as JSON
                import json

                await ws.send_str(
                    json.dumps({"text": event.token, "voice": self._opts.voice})
                )

            if event.is_flush:
                # Could send a flush signal if your server supports it
                pass

    async def _receive_audio(
        self, ws: aiohttp.ClientWebSocketResponse, output_emitter
    ) -> None:
        """Receive audio chunks from the WebSocket."""
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if msg.data:
                    output_emitter.push(msg.data)
                else:
                    # Empty binary message signals end of segment
                    output_emitter.flush()
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"[CustomTTS] WebSocket error: {ws.exception()}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break


def create_tts_with_fallback(
    primary: TTS,
    *fallbacks: TTS,
    sample_rate: int = 24000,
) -> FallbackAdapter:
    """
    Create a TTS with automatic fallback to backup providers.

    If the primary TTS fails, it will automatically switch to the fallbacks
    in order. Useful for production reliability.

    Args:
        primary: Primary TTS to use
        *fallbacks: Backup TTS instances (in priority order)
        sample_rate: Output sample rate (will resample if needed)

    Returns:
        FallbackAdapter wrapping all TTS instances

    Example:
        from tts import CustomTTS, create_tts_with_fallback
        from livekit.plugins import aws

        tts = create_tts_with_fallback(
            CustomTTS(base_url="http://gpu:8000"),  # Primary: your GPU
            aws.TTS(voice="Seoyeon"),               # Fallback: AWS Polly
        )
    """
    return FallbackAdapter(
        [primary, *fallbacks],
        sample_rate=sample_rate,
    )
