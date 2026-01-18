"""
Embedding Service - BGE-M3 (ONNX + CUDA)

사전 변환된 BGE-M3 ONNX 모델을 사용하여 GPU 가속 임베딩 생성.
HuggingFace에서 aapot/bge-m3-onnx 모델을 다운로드하여 사용.

환경변수:
    EMBEDDING_MODEL_PATH: ONNX 모델 저장 경로 (기본: /opt/models/bge-m3-onnx)
    EMBEDDING_DEVICE: 장치 (기본: cuda)
    MAX_CONCURRENT_EMBED: 최대 동시 요청 (기본: 8)

사용:
    from services.embedding_service import get_embedding_service

    service = get_embedding_service()
    service.load_model()

    vectors = await service.embed(["안녕하세요", "반갑습니다"])
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger("ai-server.embedding")

# 설정
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/opt/models/bge-m3-onnx")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
MAX_CONCURRENT_EMBED = int(os.getenv("MAX_CONCURRENT_EMBED", "8"))
MAX_SEQ_LENGTH = 512

# 사전 변환된 ONNX 모델 (HuggingFace)
PRETRAINED_ONNX_REPO = "aapot/bge-m3-onnx"


class EmbeddingService:
    """BGE-M3 임베딩 서비스 (CUDA 가속)

    사전 변환된 ONNX 모델을 CUDA ExecutionProvider로 실행하여
    빠른 임베딩 생성.

    Attributes:
        model_path: ONNX 모델 경로
        device: 실행 장치 (cuda/cpu)
        max_concurrent: 최대 동시 요청 수
        vector_dimensions: 출력 벡터 차원 (1024)
    """

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._session: Optional[ort.InferenceSession] = None
            self._tokenizer: Optional[AutoTokenizer] = None
            self._semaphore: Optional[asyncio.Semaphore] = None
            self._initialized = False
            self.model_path = EMBEDDING_MODEL_PATH
            self.device = EMBEDDING_DEVICE
            self.max_concurrent = MAX_CONCURRENT_EMBED
            self.vector_dimensions = 1024

    def _download_pretrained_onnx(self) -> None:
        """HuggingFace에서 사전 변환된 BGE-M3 ONNX 모델 다운로드"""
        logger.info(f"Downloading pre-converted BGE-M3 ONNX from {PRETRAINED_ONNX_REPO}...")

        from huggingface_hub import snapshot_download

        output_path = Path(self.model_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 모델 다운로드 (ONNX 파일과 토크나이저)
        snapshot_download(
            repo_id=PRETRAINED_ONNX_REPO,
            local_dir=str(output_path),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )

        logger.info(f"BGE-M3 ONNX model downloaded to {self.model_path}")

    def load_model(self) -> None:
        """모델 로드 (서버 시작 시 호출)"""
        if self._initialized:
            return

        model_file = Path(self.model_path) / "model.onnx"

        # ONNX 모델이 없으면 다운로드
        if not model_file.exists():
            logger.info(f"ONNX model not found at {model_file}, downloading...")
            self._download_pretrained_onnx()

        logger.info(f"Loading BGE-M3 embedding model from {self.model_path}")

        # 토크나이저 로드 (원본 BAAI/bge-m3 사용 - 호환성 보장)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception:
            logger.info("Loading tokenizer from BAAI/bge-m3")
            self._tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

        # ONNX 세션 옵션
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # ERROR only

        # Execution Provider 선택 (CUDA)
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")

        providers: list[tuple[str, dict] | str] = []

        if self.device == "cuda":
            if "CUDAExecutionProvider" in available_providers:
                providers.append(("CUDAExecutionProvider", {"device_id": 0}))
                logger.info("CUDA execution provider configured")

        providers.append("CPUExecutionProvider")

        # 세션 생성
        model_file_path = os.path.join(self.model_path, "model.onnx")
        self._session = ort.InferenceSession(
            model_file_path,
            providers=providers,
            sess_options=opts,
        )

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._initialized = True

        active_provider = self._session.get_providers()[0]
        logger.info(f"Embedding model loaded successfully (provider: {active_provider})")

    def is_loaded(self) -> bool:
        """모델 로드 여부 확인"""
        return self._initialized

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 정규화 (Cosine Similarity 필수)"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        """동기 임베딩 생성"""
        if self._tokenizer is None or self._session is None:
            raise RuntimeError("Model not loaded")

        # 토크나이즈
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )

        # ONNX 추론 - aapot/bge-m3-onnx 모델 출력 형식
        # outputs[0]: dense embeddings (batch, 1024) - 이미 [CLS] 추출됨
        # outputs[1]: sparse token weights
        # outputs[2]: colbert embeddings
        outputs = self._session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
        )

        # Dense 임베딩: 이미 (batch, 1024) 형태로 제공됨
        embeddings = outputs[0]

        # L2 정규화 (aapot 모델은 이미 정규화되어 있지만 안전을 위해)
        return self._normalize(embeddings)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """비동기 임베딩 생성

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            1024차원 벡터 리스트 (L2 정규화됨)

        Raises:
            RuntimeError: 모델이 로드되지 않은 경우
        """
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized. Call load_model() first.")

        if not texts:
            return []

        if self._semaphore is None:
            raise RuntimeError("Semaphore not initialized")

        async with self._semaphore:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._embed_sync, texts)

        return embeddings.tolist()

    async def embed_single(self, text: str) -> list[float]:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            1024차원 벡터 (L2 정규화됨)
        """
        result = await self.embed([text])
        return result[0] if result else []

    def get_status(self) -> dict:
        """서비스 상태 정보 반환"""
        return {
            "initialized": self._initialized,
            "model_path": self.model_path,
            "device": self.device,
            "max_concurrent": self.max_concurrent,
            "vector_dimensions": self.vector_dimensions,
            "provider": self._session.get_providers()[0] if self._session else None,
        }


# 전역 싱글톤
embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """임베딩 서비스 싱글톤 반환"""
    return embedding_service
