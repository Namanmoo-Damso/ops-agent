"""
로컬 ai-server 임베딩 클라이언트

ai-server의 /embed 엔드포인트를 호출하여 BGE-M3 임베딩 생성.

환경변수:
    AI_SERVER_URL: ai-server URL (기본: http://localhost:8001)
    EMBED_TIMEOUT: 요청 타임아웃 (기본: 5.0초)

사용:
    from agent.embedding_client import get_embedding_client

    client = get_embedding_client()
    vectors = await client.embed(["안녕하세요", "반갑습니다"])
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8001")
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "5.0"))


class EmbeddingClient:
    """로컬 임베딩 서버 클라이언트

    ai-server의 /embed 엔드포인트를 비동기로 호출.

    Attributes:
        base_url: ai-server 기본 URL
        timeout: 요청 타임아웃 (초)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        """클라이언트 초기화

        Args:
            base_url: ai-server URL (기본: 환경변수)
            timeout: 요청 타임아웃 (기본: 환경변수)
        """
        self.base_url = base_url or AI_SERVER_URL
        self.timeout = timeout or EMBED_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 lazy 초기화"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 임베딩 생성

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            1024차원 벡터 리스트 (L2 정규화됨)

        Raises:
            httpx.HTTPStatusError: API 오류
            httpx.RequestError: 네트워크 오류
        """
        if not texts:
            return []

        client = await self._get_client()

        try:
            response = await client.post("/embed", json={"texts": texts})
            response.raise_for_status()
            data = response.json()
            return data.get("vectors", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            1024차원 벡터 (L2 정규화됨)
        """
        result = await self.embed([text])
        return result[0] if result else []

    async def health_check(self) -> bool:
        """ai-server 상태 확인

        Returns:
            서버 정상 여부
        """
        client = await self._get_client()
        try:
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """클라이언트 연결 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None


# 전역 싱글톤
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    """임베딩 클라이언트 싱글톤 반환"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


async def close_embedding_client() -> None:
    """임베딩 클라이언트 종료"""
    global _embedding_client
    if _embedding_client:
        await _embedding_client.close()
        _embedding_client = None
