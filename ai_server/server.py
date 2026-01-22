"""
AI Server - Multi-modal AI Services API

FastAPI 기반 비동기 AI 서비스 서버
- STT (Speech-to-Text): Whisper
- TTS (Text-to-Speech): Supertonic-2
- Turn Detection: End-of-Utterance Model

GPU 메모리 최적화 및 동시 요청 제한 포함
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ai-server")

# ============================================
# Configuration
# ============================================
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "8"))

# Semaphore for request throttling
request_semaphore: asyncio.Semaphore

# Service instances (initialized in lifespan)
stt_service: Optional[Any] = None
tts_service: Optional[Any] = None
turn_detector_service: Optional[Any] = None
embedding_service: Optional[Any] = None


# ============================================
# Pydantic Models
# ============================================
class TranscriptionResponse(BaseModel):
    """STT transcription response."""

    text: str
    language: str
    language_probability: float
    duration: float
    processing_time: float
    segments: list[dict]


class TurnDetectRequest(BaseModel):
    """Turn detection request body."""

    text: str = Field(..., description="Current user utterance")
    context: Optional[list[dict]] = Field(default=None, description="Previous conversation messages")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class TurnDetectResponse(BaseModel):
    """Turn detection response."""

    end_of_turn: bool
    probability: float
    processing_time: float


class TTSRequest(BaseModel):
    """TTS synthesis request body."""

    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="F2", description="Voice style (M1-M5, F1-F5)")
    speed: float = Field(default=1.0, ge=0.7, le=2.0)
    language: Optional[str] = Field(default=None, description="Language code (ko, en, etc.)")


class TTSResponse(BaseModel):
    """TTS synthesis response metadata."""

    duration: float
    processing_time: float
    voice: str
    language: str
    sample_rate: int


class EmbedRequest(BaseModel):
    """Text embedding request body."""

    texts: list[str] = Field(..., description="List of texts to embed")


class EmbedResponse(BaseModel):
    """Text embedding response."""

    vectors: list[list[float]] = Field(..., description="1024-dim L2-normalized vectors")
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    stt_loaded: bool
    tts_loaded: bool
    turn_detector_loaded: bool
    embedding_loaded: bool
    concurrent_limit: int


# ============================================
# Lifespan - Model Loading
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management - load models on startup."""
    global request_semaphore, stt_service, tts_service, turn_detector_service, embedding_service

    logger.info("=" * 50)
    logger.info("Starting AI Server...")
    logger.info("=" * 50)

    # Initialize semaphore
    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")

    # Load STT service (Whisper)
    try:
        from services.stt_service import STTService
        stt_service = STTService()
        await stt_service.initialize()
        logger.info("STT Service loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load STT service: {e}")

    # Load TTS service (Supertonic-2) - optional, disabled by default
    if os.getenv("ENABLE_TTS", "false").lower() == "true":
        try:
            from services.tts_service import TTSService
            tts_service = TTSService()
            await tts_service.initialize()
            logger.info("TTS Service loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load TTS service: {e}")
    else:
        logger.info("TTS Service disabled (ENABLE_TTS != true)")

    # Load Turn Detector service
    try:
        from services.turn_detector_service import TurnDetectorService
        turn_detector_service = TurnDetectorService()
        turn_detector_service.load_model()
        logger.info("Turn Detector Service loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Turn Detector service: {e}")

    # Load Embedding service (BGE-M3, TensorRT)
    try:
        from services.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        embedding_service.load_model()
        logger.info("Embedding Service loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Embedding service: {e}")

    logger.info("=" * 50)
    logger.info("AI Server started successfully")
    logger.info("=" * 50)

    yield

    # Cleanup
    logger.info("Shutting down AI Server...")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="AI Server - Multi-modal AI Services",
    description="STT, TTS, and Turn Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================
# Helper Functions
# ============================================
# (Moved to services/stt_service.py)


# ============================================
# API Endpoints
# ============================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        stt_loaded=stt_service.is_initialized() if stt_service else False,
        tts_loaded=tts_service.is_initialized() if tts_service else False,
        turn_detector_loaded=turn_detector_service._initialized if turn_detector_service else False,
        embedding_loaded=embedding_service.is_loaded() if embedding_service else False,
        concurrent_limit=MAX_CONCURRENT_REQUESTS,
    )


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
):
    """
    OpenAI Whisper API compatible endpoint.

    - file: Audio file (WAV, MP3, etc.)
    - language: Language code (optional, e.g., "ko", "en")
    """
    start_time = time.time()

    if not stt_service or not stt_service.is_initialized():
        raise HTTPException(status_code=503, detail="STT service not available")

    async with request_semaphore:
        try:
            audio_bytes = await file.read()

            if len(audio_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty audio file")

            result = await stt_service.transcribe(audio_bytes, language=language)
            processing_time = time.time() - start_time

            return TranscriptionResponse(
                text=result["text"],
                language=result["language"],
                language_probability=result["language_probability"],
                duration=result["duration"],
                processing_time=processing_time,
                segments=result["segments"],
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/v1/audio/speech")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form(default="F2"),
    speed: float = Form(default=1.0),
    language: Optional[str] = Form(default=None),
    output_format: str = Form(default="wav"),
):
    """
    Text-to-Speech endpoint (Supertonic-2).

    - text: Text to synthesize
    - voice: Voice style (M1-M5: male, F1-F5: female)
    - speed: Speaking speed (0.7-2.0)
    - language: Language code (ko, en, etc.), auto-detected if not provided
    - output_format: Output format (wav or pcm)
    """
    start_time = time.time()

    if not tts_service or not tts_service.is_initialized():
        raise HTTPException(status_code=503, detail="TTS service not available")

    async with request_semaphore:
        try:
            audio_bytes = await tts_service.synthesize(
                text=text,
                voice=voice,
                speed=speed,
                language=language,
                output_format=output_format,
            )
            processing_time = time.time() - start_time

            media_type = "audio/wav" if output_format == "wav" else "audio/pcm"

            return StreamingResponse(
                io.BytesIO(audio_bytes),
                media_type=media_type,
                headers={
                    "X-Processing-Time": str(processing_time),
                    "X-Sample-Rate": str(tts_service.sample_rate),
                    "X-Voice": voice,
                },
            )

        except Exception as e:
            logger.exception(f"TTS error: {e}")
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.post("/v1/stt/raw")
async def transcribe_raw(
    audio: bytes,
    sample_rate: int = 16000,
    language: Optional[str] = None,
):
    """
    Raw PCM transcription endpoint.

    Direct processing for LiveKit Agent integration.
    """
    start_time = time.time()

    if not stt_service or not stt_service.is_initialized():
        raise HTTPException(status_code=503, detail="STT service not available")

    async with request_semaphore:
        try:
            result = await stt_service.transcribe(
                audio_data=audio,
                language=language,
                sample_rate=sample_rate,
            )
            processing_time = time.time() - start_time

            return {
                "text": result["text"],
                "language": result["language"],
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.exception(f"Raw transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Turn Detection Endpoint
# ============================================
@app.post("/v1/turn/detect")
async def detect_turn(request: TurnDetectRequest):
    """
    Turn detection endpoint.

    Predicts the probability that the user has finished their turn.

    - text: Current user utterance
    - context: Previous conversation messages (optional)
    - threshold: Probability threshold for end-of-turn decision
    """
    start_time = time.time()

    if not turn_detector_service or not turn_detector_service._initialized:
        raise HTTPException(status_code=503, detail="Turn Detector service not available")

    try:
        result = await turn_detector_service.detect_end_of_turn(
            text=request.text,
            context=request.context,
            threshold=request.threshold,
        )
        processing_time = time.time() - start_time

        logger.info(
            f"[TurnDetector GPU] inference={processing_time*1000:.2f}ms "
            f"end_of_turn={result.end_of_turn} prob={result.probability:.3f}"
        )

        return TurnDetectResponse(
            end_of_turn=result.end_of_turn,
            probability=result.probability,
            processing_time=processing_time,
        )

    except Exception as e:
        logger.exception(f"Turn detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Turn detection failed: {str(e)}")


# ============================================
# Embedding Endpoint
# ============================================
@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest) -> EmbedResponse:
    """
    Text embedding endpoint (BGE-M3, TensorRT).

    Generates 1024-dimensional L2-normalized vectors for semantic search.

    - texts: List of texts to embed (max 512 tokens per text)

    Returns:
        vectors: List of 1024-dim vectors (L2-normalized)
        processing_time: Inference time in seconds
    """
    start_time = time.time()

    if not embedding_service or not embedding_service.is_loaded():
        raise HTTPException(status_code=503, detail="Embedding service not available")

    if not request.texts:
        return EmbedResponse(vectors=[], processing_time=0.0)

    async with request_semaphore:
        try:
            vectors = await embedding_service.embed(request.texts)
            processing_time = time.time() - start_time

            logger.info(
                f"[Embedding] texts={len(request.texts)} "
                f"inference={processing_time*1000:.2f}ms"
            )

            return EmbedResponse(
                vectors=vectors,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.exception(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


# ============================================
# Main Entry Point
# ============================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("AI_SERVER_PORT", "8001"))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # GPU models require single worker
        log_level="info",
    )
