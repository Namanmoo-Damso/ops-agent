"""
RAG Client for Python Agent

Provides utilities to search conversation history and get relevant context
from PGVector storage (Bedrock Titan Embeddings V2 - 1024 dimensions)
"""
import os
import logging
from typing import List, Dict, Any, Optional
import httpx

from constants import (
    TIMEOUT_CALL_CONTEXT,
)

logger = logging.getLogger(__name__)
_shared_rag_client: Optional["RagClient"] = None


class RagClient:
    """
    Client for interacting with RAG (Retrieval-Augmented Generation) service

    Usage:
        rag_client = RagClient()

        # Search for similar conversations
        results = await rag_client.search_similar(
            ward_id="uuid",
            query="건강 상태에 대해 어떻게 말했나요?",
            limit=5
        )

        # Get recent conversation context
        context = await rag_client.get_recent_context(
            ward_id="uuid",
            limit=10
        )
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: float = TIMEOUT_CALL_CONTEXT,  # Default aligned with shared timeout config
    ):
        """
        Initialize RAG client

        Args:
            api_base: Base URL for API (defaults to API_BASE_URL env var)
            api_token: Authentication token (defaults to API_INTERNAL_TOKEN env var, REQUIRED)
            timeout: Request timeout in seconds
        """
        self.api_base = api_base or os.getenv("API_BASE_URL", "http://localhost:3000")
        self.api_token = api_token or os.getenv("API_INTERNAL_TOKEN")
        self.timeout = float(timeout)

        # Ensure trailing slash is removed
        self.api_base = self.api_base.rstrip("/")

        # Validate that API token is available
        if not self.api_token:
            logger.warning(
                "⚠️  API_INTERNAL_TOKEN not set - RAG requests may fail with 401 Unauthorized. "
                "Set API_INTERNAL_TOKEN environment variable for authentication."
            )

        logger.info(f"RAG Client initialized: {self.api_base} (timeout={timeout}s)")

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def search_similar(
        self,
        ward_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar conversation chunks

        Args:
            ward_id: Ward UUID to search within
            query: Natural language query to search for
            limit: Maximum number of results to return

        Returns:
            List of results with text, metadata, and similarity score
            Example:
            [
                {
                    "text": "[user]: 요즘 무릎이 아파요\n[assistant]: 무릎 통증이 있으시군요...",
                    "metadata": {
                        "speakers": ["user", "assistant"],
                        "timestamp": "2024-01-10T12:00:00Z",
                        "chunkLength": 250
                    },
                    "similarity": 0.85
                }
            ]
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/v1/rag/search",
                    params={
                        "wardId": ward_id,
                        "query": query,
                        "limit": limit,
                    },
                    headers=self._get_headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
        except httpx.TimeoutException:
            # Timeout is common with vector search - log and return empty
            logger.warning(f"RAG search timed out after {self.timeout}s for ward={ward_id}")
            return []
        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx) - log status code
            logger.error(f"RAG search HTTP error {e.response.status_code}: {e}")
            return []
        except httpx.HTTPError as e:
            # Network/connection errors
            logger.error(f"RAG search network error: {e}")
            return []
        except Exception as e:
            # Unexpected errors - log with stack trace
            logger.error(f"Unexpected error during RAG search: {e}", exc_info=True)
            return []

    async def get_recent_context(
        self,
        ward_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for a ward

        Args:
            ward_id: Ward UUID
            limit: Maximum number of conversation chunks to return

        Returns:
            List of recent conversation chunks ordered by time (newest first)
            Example:
            [
                {
                    "text": "Recent conversation...",
                    "createdAt": "2024-01-10T12:00:00Z"
                }
            ]
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base}/v1/rag/context",
                    params={
                        "wardId": ward_id,
                        "limit": limit,
                    },
                    headers=self._get_headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("context", [])
        except httpx.TimeoutException:
            # Timeout - return empty to avoid blocking caller
            logger.warning(f"Get context timed out after {self.timeout}s for ward={ward_id}")
            return []
        except httpx.HTTPStatusError as e:
            # HTTP error - log status code
            logger.error(f"Get context HTTP error {e.response.status_code}: {e}")
            return []
        except httpx.HTTPError as e:
            # Network/connection errors
            logger.error(f"Failed to get context (network): {e}")
            return []
        except Exception as e:
            # Unexpected errors - log with stack trace
            logger.error(f"Unexpected error getting context: {e}", exc_info=True)
            return []

    async def build_context_prompt(
        self,
        ward_id: str,
        query: str,
        search_limit: int = 3,
        context_limit: int = 5,
    ) -> str:
        """
        Build a context-aware prompt by combining relevant past conversations

        This is useful for giving the AI agent context about previous conversations

        Args:
            ward_id: Ward UUID
            query: Current query or conversation topic
            search_limit: Number of similar conversations to search
            context_limit: Number of recent conversations to include

        Returns:
            Formatted context string that can be added to AI prompts
        """
        try:
            # Get both similar and recent contexts
            # These calls have their own timeout/error handling
            similar_results = await self.search_similar(ward_id, query, search_limit)
            recent_context = await self.get_recent_context(ward_id, context_limit)

            context_parts = []

            # Add similar conversations
            # SAFE: Use .get() to avoid KeyError if API response missing fields
            if similar_results:
                context_parts.append("=== 관련 과거 대화 ===")
                for i, result in enumerate(similar_results, 1):
                    # Safe access: default to 0 if similarity missing
                    similarity_pct = int(result.get("similarity", 0) * 100)
                    # Safe access: skip if text missing (avoid KeyError)
                    text = result.get("text", "")
                    if text:  # Only add if text exists
                        context_parts.append(
                            f"\n[관련도 {similarity_pct}%]\n{text}\n"
                        )

            # Add recent context
            # SAFE: Use .get() to handle missing 'text' field gracefully
            if recent_context:
                context_parts.append("\n=== 최근 대화 내역 ===")
                for ctx in recent_context:
                    # Safe access: skip if text missing
                    text = ctx.get("text", "")
                    if text:  # Only add if text exists
                        context_parts.append(f"\n{text}\n")

            # Return joined parts or fallback message
            if context_parts:
                return "\n".join(context_parts)
            else:
                return "이전 대화 기록이 없습니다."

        except Exception as e:
            # Log error with stack trace for debugging
            logger.error(f"Failed to build context prompt: {e}", exc_info=True)
            # Return empty string to avoid breaking caller
            return ""


def get_shared_rag_client(timeout: Optional[float] = None) -> "RagClient":
    """
    Get a shared RagClient instance to avoid recreating clients per request.

    Args:
        timeout: Optional timeout override; recreates the shared client if different.
    """
    global _shared_rag_client
    if _shared_rag_client is None:
        _shared_rag_client = RagClient(timeout=timeout) if timeout is not None else RagClient()
    elif timeout is not None and _shared_rag_client.timeout != float(timeout):
        _shared_rag_client = RagClient(timeout=timeout)
    return _shared_rag_client


# Convenience function for quick usage
async def search_conversation_history(
    ward_id: str,
    query: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience function to quickly search conversation history

    Args:
        ward_id: Ward UUID
        query: Search query
        limit: Maximum results

    Returns:
        List of matching conversation chunks
    """
    client = get_shared_rag_client()
    return await client.search_similar(ward_id, query, limit)
