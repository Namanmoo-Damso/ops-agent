"""
SearchThresholdFilter - RAG 검색 결과 필터링

역할: FTS 결과가 우수할 경우 벡터 점수가 낮아도 통과시키는 Safe-pass 로직
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SearchThresholdFilter:
    """검색 결과 필터링 - FTS Safe-pass 지원."""

    def __init__(
        self,
        min_vector_score: float = 0.3,
        fts_safe_pass_score: float = 0.6,
    ):
        """
        Args:
            min_vector_score: 최소 벡터 유사도 (기본 0.3)
            fts_safe_pass_score: FTS 점수가 이 이상이면 Safe-pass (기본 0.6)
        """
        self.min_vector_score = min_vector_score
        self.fts_safe_pass_score = fts_safe_pass_score

    def filter_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        검색 결과 필터링.

        - 벡터 유사도 >= min_vector_score: 통과
        - FTS rank_score >= fts_safe_pass_score: Safe-pass (벡터 점수 무시)

        Args:
            results: RAG 검색 결과 리스트

        Returns:
            필터링된 결과 리스트
        """
        if not results:
            return []

        filtered = []
        for result in results:
            similarity = result.get("similarity", 0)
            
            # FTS rank_score가 높으면 Safe-pass
            metadata = result.get("metadata", {})
            chunk_header = metadata.get("chunk_header", "")
            
            # similarity가 실제로 FTS rank_score인 경우 (하이브리드 검색)
            if similarity >= self.fts_safe_pass_score:
                filtered.append(result)
                logger.debug(f"Safe-pass: score={similarity:.3f}")
            elif similarity >= self.min_vector_score:
                filtered.append(result)
                logger.debug(f"Pass: score={similarity:.3f}")
            else:
                logger.debug(f"Filtered out: score={similarity:.3f}")

        logger.info(
            f"[Filter] {len(results)} -> {len(filtered)} results "
            f"(min={self.min_vector_score}, safe_pass={self.fts_safe_pass_score})"
        )
        return filtered
