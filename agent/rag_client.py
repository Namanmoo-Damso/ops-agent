"""
RAG Client for Python Agent

Provides utilities to search conversation history and get relevant context
from PGVector storage (Bedrock Titan Embeddings V2 - 1024 dimensions)
"""
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx

from constants import (
    AGENT_TZINFO,
    TIMEOUT_CALL_CONTEXT,
)

logger = logging.getLogger(__name__)
_shared_rag_client: Optional["RagClient"] = None


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_months_ago(months: int) -> str:
    if months == 1:
        return "한 달 전"
    if months == 2:
        return "두 달 전"
    return f"{months}개월 전"


def format_relative_time_ko(
    iso_timestamp: str,
    now: Optional[datetime] = None,
) -> Optional[str]:
    dt = _parse_iso_datetime(iso_timestamp)
    if not dt:
        return None

    tz = AGENT_TZINFO or timezone.utc
    dt_local = dt.astimezone(tz)
    now_local = now.astimezone(tz) if now else datetime.now(tz)

    if now_local < dt_local:
        return "방금 전"

    diff = now_local - dt_local
    diff_seconds = int(diff.total_seconds())
    diff_hours = diff_seconds // 3600
    diff_days = diff_seconds // 86400

    if diff_hours < 1:
        return "방금 전"
    if diff_hours < 24:
        return f"{diff_hours}시간 전"
    if diff_days == 1:
        return "어제"
    if diff_days < 7:
        return f"{diff_days}일 전"
    if diff_days < 14:
        return "지난주"
    if diff_days < 60:
        weeks = diff_days // 7
        return f"{weeks}주 전"

    months = diff_days // 30
    return _format_months_ago(months)


def extract_result_text(result: Dict[str, Any]) -> str:
    """Select the most relevant text field from a RAG search result."""
    snippet = result.get("snippet", "")
    parent_text = result.get("parentText", "")
    return snippet or parent_text or result.get("text", "")


def extract_result_time_label(result: Dict[str, Any]) -> Optional[str]:
    created_at = result.get("createdAt", "")
    return format_relative_time_ko(created_at)


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
        Search for similar conversation chunks using Parent-Child structure

        Args:
            ward_id: Ward UUID to search within
            query: Natural language query to search for
            limit: Maximum number of results to return

        Returns:
            List of results with parent context and snippet
            Example:
            [
                {
                    "text": "...무릎이 아파요...",  # Window snippet around match
                    "childText": "무릎이 아파요",
                    "parentText": "[날짜: 2024-01-10] [user]: 요즘 무릎이 아파요\n[assistant]: ...",
                    "parentId": "uuid",
                    "snippet": "...무릎이 아파요...",
                    "metadata": {...},
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

    async def preload_weekly_context(
        self,
        ward_id: str,
        call_direction: str = "inbound",
    ) -> bool:
        """
        Preload weekly context into Redis cache before call starts

        This should be called when participants join the room (before first greeting)
        to ensure fast RAG searches during the conversation.

        Args:
            ward_id: Ward UUID
            call_direction: Call direction ("inbound" or "outbound")

        Returns:
            True if preload started successfully, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/v1/rag/preload",
                    json={"wardId": ward_id, "callDirection": call_direction},
                    headers=self._get_headers(),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                logger.info(f"✅ Weekly context preload started for ward={ward_id}, direction={call_direction}")
                return True
        except httpx.TimeoutException:
            logger.warning(f"⚠️  Preload timed out after {self.timeout}s for ward={ward_id}")
            return False
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Preload HTTP error {e.response.status_code}: {e}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"❌ Preload network error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during preload: {e}", exc_info=True)
            return False

    async def build_context_prompt(
        self,
        ward_id: str,
        query: str,
        search_limit: int = 3,
        context_limit: int = 5,
    ) -> str:
        """
        Build a context-aware prompt using Parent-Child structure

        Now uses parent context for broader understanding and snippet for focused relevance
        Includes relative time hints in Asia/Seoul timezone for better time awareness

        Args:
            ward_id: Ward UUID
            query: Current query or conversation topic
            search_limit: Number of similar conversations to search
            context_limit: Number of recent conversations to include

        Returns:
            Formatted context string with parent context and timestamps
        """
        try:
            # Get both similar and recent contexts
            # These calls have their own timeout/error handling
            similar_results = await self.search_similar(ward_id, query, search_limit)
            recent_context = await self.get_recent_context(ward_id, context_limit)

            context_parts = []

            # Add similar conversations with parent context and timestamps
            if similar_results:
                context_parts.append("=== 관련 과거 대화 ===")
                for i, result in enumerate(similar_results, 1):
                    # Safe access: default to 0 if similarity missing
                    similarity_pct = int(result.get("similarity", 0) * 100)

                    text = extract_result_text(result)
                    relative_time = extract_result_time_label(result)

                    if text:  # Only add if text exists
                        time_label = (
                            f" · {relative_time}" if relative_time else ""
                        )
                        context_parts.append(
                            f"\n[관련도 {similarity_pct}%{time_label}]\n{text}\n"
                        )

            # Add recent context with timestamps
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
