"""
Turn Detector Service - End-of-Utterance Detection using ONNX (GPU)

Based on livekit-plugins-turn-detector MultilingualModel implementation.
Provides turn detection with GPU acceleration via onnxruntime-gpu.
"""

import asyncio
import json
import logging
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime

logger = logging.getLogger("ai-server.turn-detector")

# Configuration
TURN_DETECTOR_DEVICE = os.getenv("TURN_DETECTOR_DEVICE", "cuda")  # cuda or cpu
TURN_DETECTOR_MODEL_PATH = os.getenv("TURN_DETECTOR_MODEL_PATH")  # Optional custom model path

# Model constants (from livekit-plugins-turn-detector v1.3.x)
HG_MODEL = "livekit/turn-detector"
MODEL_REVISION = "v0.4.1-intl"  # multilingual revision (updated)
ONNX_FILENAME = "model_q8.onnx"  # quantized model filename
MAX_HISTORY_TOKENS = 128
MAX_HISTORY_TURNS = 6


@dataclass
class TurnDetectionResult:
    """Turn detection result."""

    end_of_turn: bool
    probability: float
    processing_time: float
    input_text: str


class TurnDetectorService:
    """
    Turn Detector Service using ONNX runtime with GPU support.

    This model predicts the probability that the user has finished speaking
    based on the conversation context.
    """

    _instance: Optional["TurnDetectorService"] = None
    _session: Optional[onnxruntime.InferenceSession] = None
    _tokenizer: Any = None
    _languages: dict[str, Any] = {}
    _initialized: bool = False

    def __new__(cls) -> "TurnDetectorService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()

    def load_model(self) -> None:
        """
        Load the Turn Detector ONNX model.

        Should be called during server startup (lifespan).
        """
        if self._initialized:
            return

        logger.info(f"Loading Turn Detector model (device: {TURN_DETECTOR_DEVICE})")

        # Determine model path
        model_path = TURN_DETECTOR_MODEL_PATH
        languages_path = None

        if not model_path:
            try:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(
                    repo_id=HG_MODEL,
                    filename=ONNX_FILENAME,
                    subfolder="onnx",
                    revision=MODEL_REVISION,
                )
                languages_path = hf_hub_download(
                    repo_id=HG_MODEL,
                    filename="languages.json",
                    revision=MODEL_REVISION,
                )
                logger.info(f"Downloaded model from HuggingFace: {model_path}")
            except Exception as e:
                raise RuntimeError(f"Could not download Turn Detector model: {e}") from e

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Turn Detector model not found: {model_path}")

        # Load languages config
        if languages_path and Path(languages_path).exists():
            with open(languages_path) as f:
                self._languages = json.load(f)
            logger.info(f"Loaded {len(self._languages)} language configurations")

        # Load tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL,
                revision=MODEL_REVISION,
                truncation_side="left",
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Could not load tokenizer: {e}") from e

        # Configure session options
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = max(1, min(math.ceil(os.cpu_count() or 4) // 2, 4))
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.dynamic_block_base", "4")

        # Select execution provider
        available_providers = onnxruntime.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")

        if TURN_DETECTOR_DEVICE == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
            logger.info("Using CUDA execution provider for Turn Detector")
        else:
            providers = ["CPUExecutionProvider"]
            logger.info("Using CPU execution provider for Turn Detector")

        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
            sess_options=opts,
        )
        self._initialized = True
        logger.info("Turn Detector model loaded successfully")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for turn detection."""
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text.lower())
        # Remove punctuation except apostrophe and hyphen
        text = "".join(
            ch
            for ch in text
            if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"])
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _format_chat_context(self, messages: list[dict[str, str]]) -> str:
        """
        Format chat context for the model.

        Args:
            messages: List of messages with 'role' and 'content' keys

        Returns:
            Formatted conversation text
        """
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")

        # Combine adjacent turns and normalize
        new_chat_ctx: list[dict[str, str]] = []
        last_msg: Optional[dict[str, str]] = None

        for msg in messages:
            if not msg.get("content"):
                continue

            content = self._normalize_text(msg["content"])
            role = msg.get("role", "user")

            if last_msg and last_msg["role"] == role:
                last_msg["content"] += f" {content}"
            else:
                new_msg = {"role": role, "content": content}
                new_chat_ctx.append(new_msg)
                last_msg = new_msg

        # Apply chat template
        convo_text = self._tokenizer.apply_chat_template(
            new_chat_ctx,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )

        # Remove the EOU token from current utterance
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix] if ix >= 0 else convo_text

        return text

    def get_unlikely_threshold(self, language: str | None) -> float | None:
        """
        Get the unlikely threshold for a language.

        Below this threshold, the model is uncertain about end-of-turn.
        """
        if not language:
            return None

        lang = language.lower()
        lang_data = self._languages.get(lang)

        # Try base language if full code not found
        if lang_data is None and "-" in lang:
            base_lang = lang.split("-")[0]
            lang_data = self._languages.get(base_lang)

        if lang_data:
            return lang_data.get("threshold")

        return None

    def supports_language(self, language: str | None) -> bool:
        """Check if the model supports a language."""
        return self.get_unlikely_threshold(language) is not None

    async def detect_end_of_turn(
        self,
        text: str,
        context: list[dict[str, str]] | None = None,
        threshold: float = 0.5,
    ) -> TurnDetectionResult:
        """
        Detect if the user has finished their turn.

        Args:
            text: Current user utterance
            context: Previous conversation messages (optional)
            threshold: Probability threshold for end-of-turn decision

        Returns:
            TurnDetectionResult with end_of_turn flag and probability
        """
        start_time = time.time()

        if self._session is None:
            raise RuntimeError("Turn Detector model not loaded. Call load_model() first.")

        # Build message list
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context[-MAX_HISTORY_TURNS:])

        # Add current user message
        messages.append({"role": "user", "content": text})

        loop = asyncio.get_event_loop()

        async with self._lock:
            # Format and tokenize
            formatted_text = await loop.run_in_executor(
                None, self._format_chat_context, messages
            )

            def _run_inference() -> float:
                inputs = self._tokenizer(
                    formatted_text,
                    add_special_tokens=False,
                    return_tensors="np",
                    max_length=MAX_HISTORY_TOKENS,
                    truncation=True,
                )

                # Run inference
                outputs = self._session.run(
                    None,
                    {"input_ids": inputs["input_ids"].astype("int64")}
                )

                # Get probability from last position
                eou_probability = outputs[0].flatten()[-1]
                return float(eou_probability)

            probability = await loop.run_in_executor(None, _run_inference)

        processing_time = time.time() - start_time

        return TurnDetectionResult(
            end_of_turn=probability >= threshold,
            probability=probability,
            processing_time=processing_time,
            input_text=formatted_text[:200] + "..." if len(formatted_text) > 200 else formatted_text,
        )

    async def predict_end_of_turn_batch(
        self,
        requests: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> list[TurnDetectionResult]:
        """
        Batch prediction for multiple requests.

        Args:
            requests: List of dicts with 'text' and optional 'context' keys
            threshold: Probability threshold for end-of-turn decision

        Returns:
            List of TurnDetectionResult
        """
        # For now, process sequentially (can be optimized for batching later)
        results = []
        for req in requests:
            result = await self.detect_end_of_turn(
                text=req.get("text", ""),
                context=req.get("context"),
                threshold=threshold,
            )
            results.append(result)
        return results


# Global singleton instance
turn_detector_service = TurnDetectorService()


def get_turn_detector_service() -> TurnDetectorService:
    """Get the Turn Detector service singleton."""
    return turn_detector_service
