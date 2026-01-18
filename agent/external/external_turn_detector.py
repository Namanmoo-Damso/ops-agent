"""
External Turn Detector implementation for LiveKit Agents.

HTTP-based turn detector that calls AI Server for end-of-turn prediction.
Turn detection inference runs on GPU, Agent only sends text and receives predictions.
"""

import asyncio
import json
import logging
from typing import Optional

import aiohttp
from livekit.agents import llm

logger = logging.getLogger(__name__)

# Default threshold values (from multilingual model)
DEFAULT_UNLIKELY_THRESHOLD = 0.1
SUPPORTED_LANGUAGES = {
    "ko": 0.1,  # Korean
    "en": 0.15,  # English
    "ja": 0.1,  # Japanese
    "zh": 0.1,  # Chinese
    "es": 0.15,  # Spanish
    "fr": 0.15,  # French
    "de": 0.15,  # German
}


class ExternalTurnDetector:
    """
    External Turn Detector that offloads inference to AI Server GPU.

    Implements the TurnDetector protocol for LiveKit Agents.
    Uses HTTP POST to /v1/turn/detect endpoint on AI Server.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 5.0,
        default_threshold: float = 0.5,
    ):
        """
        Initialize External Turn Detector.

        Args:
            base_url: AI Server base URL (e.g., "http://localhost:8001")
            timeout: Request timeout in seconds
            default_threshold: Default probability threshold for end-of-turn
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_threshold = default_threshold
        self._http_session: Optional[aiohttp.ClientSession] = None

        logger.info(f"ExternalTurnDetector initialized: {self._base_url}")

    @property
    def model(self) -> str:
        return "multilingual-turn-detector-gpu"

    @property
    def provider(self) -> str:
        return "external-ai-server"

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout, connect=2.0)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def unlikely_threshold(self, language: str | None) -> float | None:
        """
        Get the unlikely threshold for a language.

        Below this threshold, the model is uncertain about end-of-turn.
        """
        if not language:
            return DEFAULT_UNLIKELY_THRESHOLD

        lang = language.lower()
        threshold = SUPPORTED_LANGUAGES.get(lang)

        # Try base language if full code not found
        if threshold is None and "-" in lang:
            base_lang = lang.split("-")[0]
            threshold = SUPPORTED_LANGUAGES.get(base_lang)

        return threshold

    async def supports_language(self, language: str | None) -> bool:
        """Check if the model supports a language."""
        threshold = await self.unlikely_threshold(language)
        return threshold is not None

    async def predict_end_of_turn(
        self,
        chat_ctx: llm.ChatContext,
        *,
        timeout: float | None = None,
    ) -> float:
        """
        Predict the probability that the user has finished their turn.

        Args:
            chat_ctx: Current conversation context
            timeout: Optional timeout override

        Returns:
            Probability (0.0 to 1.0) that the turn is complete
        """
        # Format chat context for API
        messages = []
        for msg in chat_ctx.items:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = str(msg.role) if hasattr(msg.role, "value") else msg.role
                content = ""

                # Handle different content types
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content parts
                    for part in msg.content:
                        if hasattr(part, "text"):
                            content += part.text
                        elif isinstance(part, str):
                            content += part

                if content:
                    messages.append({"role": role, "content": content})

        if not messages:
            logger.debug("Empty chat context, returning 0.0")
            return 0.0

        # Get the last user message for turn detection
        last_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") in ("user", "human"):
                last_user_message = msg.get("content", "")
                break

        if not last_user_message:
            logger.debug("No user message in context, returning 0.0")
            return 0.0

        try:
            session = await self._ensure_session()

            # Call AI Server turn detection endpoint
            url = f"{self._base_url}/v1/turn/detect"

            # Prepare request body
            request_data = {
                "text": last_user_message,
                "context": messages[-6:],  # Keep last 6 messages for context
                "threshold": self._default_threshold,
            }

            request_timeout = timeout or self._timeout
            import time
            start_time = time.time()

            async with session.post(
                url,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.warning(
                        f"Turn detection API error: {response.status} - {error_text}"
                    )
                    # Fall back to simple heuristic
                    return self._simple_heuristic(last_user_message)

                result = await response.json()
                probability = result.get("probability", 0.0)
                end_of_turn = result.get("end_of_turn", False)
                server_processing_time = result.get("processing_time", 0.0)

                total_time_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"[ExternalTurnDetector] total={total_time_ms:.2f}ms "
                    f"server={server_processing_time*1000:.2f}ms "
                    f"end_of_turn={end_of_turn} prob={probability:.3f}"
                )
                return float(probability)

        except asyncio.TimeoutError:
            logger.warning(f"Turn detection timed out after {request_timeout}s")
            return self._simple_heuristic(last_user_message)
        except aiohttp.ClientError as e:
            logger.warning(f"Turn detection HTTP error: {e}")
            return self._simple_heuristic(last_user_message)
        except Exception as e:
            logger.error(f"Turn detection error: {e}")
            return self._simple_heuristic(last_user_message)

    def _simple_heuristic(self, text: str) -> float:
        """
        Simple heuristic for end-of-turn when server is unavailable.

        Returns higher probability for:
        - Questions (ending with ?)
        - Sentences ending with period
        - Exclamations
        """
        text = text.strip()
        if not text:
            return 0.0

        # Check for sentence-ending punctuation
        if text.endswith(("?", "？")):
            return 0.9  # Questions are likely end of turn
        if text.endswith((".", "。", "!", "！")):
            return 0.7  # Statements/exclamations
        if text.endswith(("요", "다", "죠", "네")):
            # Korean sentence endings
            return 0.6

        # Default: uncertain
        return 0.3

    async def aclose(self) -> None:
        """Clean up HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
