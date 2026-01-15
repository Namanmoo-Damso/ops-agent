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

        통과 조건 (우선순위 순):
        1. hasKeywordMatch == True: FTS 키워드 매칭 Safe-pass
        2. isRecommendation == True: FTS 추천 결과 Safe-pass
        3. similarity >= fts_safe_pass_score: 높은 점수 Safe-pass
        4. similarity >= min_vector_score: 벡터 유사도 통과

        Args:
            results: RAG 검색 결과 리스트

        Returns:
            필터링된 결과 리스트
        """
        if not results:
            return []

        filtered = []
        metadata_not_dict = 0
        missing_keyword_match = 0
        missing_recommendation = 0
        invalid_keyword_match = 0
        invalid_recommendation = 0
        sample_indices: list[int] = []

        def record_issue(index: int) -> None:
            if len(sample_indices) < 5:
                sample_indices.append(index)

        def normalize_flag(
            metadata_obj: dict[str, Any],
            key: str,
            index: int,
        ) -> tuple[bool, bool]:
            nonlocal missing_keyword_match, missing_recommendation
            nonlocal invalid_keyword_match, invalid_recommendation

            if key not in metadata_obj:
                if key == "hasKeywordMatch":
                    missing_keyword_match += 1
                else:
                    missing_recommendation += 1
                record_issue(index)
                return False, True

            value = metadata_obj.get(key)
            if isinstance(value, bool):
                return value, False

            if isinstance(value, int) and value in (0, 1):
                if key == "hasKeywordMatch":
                    invalid_keyword_match += 1
                else:
                    invalid_recommendation += 1
                record_issue(index)
                return bool(value), True

            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in ("true", "false"):
                    if key == "hasKeywordMatch":
                        invalid_keyword_match += 1
                    else:
                        invalid_recommendation += 1
                    record_issue(index)
                    return normalized == "true", True

            if key == "hasKeywordMatch":
                invalid_keyword_match += 1
            else:
                invalid_recommendation += 1
            record_issue(index)
            return False, True

        for index, result in enumerate(results, 1):
            similarity = result.get("similarity", 0)
            metadata = result.get("metadata", {})
            metadata_issue = False

            if not isinstance(metadata, dict):
                metadata_not_dict += 1
                record_issue(index)
                metadata = {}
                metadata_issue = True
            
            # 키워드 매칭 플래그 확인 (API에서 RRF fusion 시 설정)
            has_keyword_match, keyword_issue = normalize_flag(
                metadata, "hasKeywordMatch", index
            )
            is_recommendation, recommendation_issue = normalize_flag(
                metadata, "isRecommendation", index
            )
            metadata_issue = metadata_issue or keyword_issue or recommendation_issue
            
            # 1. FTS 키워드 매칭 결과: 점수와 상관없이 Safe-pass
            if has_keyword_match:
                filtered.append(result)
                logger.debug(f"Safe-pass (keyword match): score={similarity:.4f}")
            # 2. FTS 추천 결과: Safe-pass
            elif is_recommendation:
                filtered.append(result)
                logger.debug(f"Safe-pass (recommendation): score={similarity:.4f}")
            # 3. 높은 점수: Safe-pass
            elif similarity >= self.fts_safe_pass_score:
                filtered.append(result)
                logger.debug(f"Safe-pass (high score): score={similarity:.3f}")
            # 4. 벡터 유사도 임계값 이상
            elif similarity >= self.min_vector_score:
                filtered.append(result)
                logger.debug(f"Pass (vector): score={similarity:.3f}")
            elif metadata_issue and similarity > 0:
                filtered.append(result)
                logger.debug(
                    f"Safe-pass (metadata issue): score={similarity:.4f}"
                )
            else:
                logger.debug(f"Filtered out: score={similarity:.4f}")

        if (
            metadata_not_dict
            or missing_keyword_match
            or missing_recommendation
            or invalid_keyword_match
            or invalid_recommendation
        ):
            logger.warning(
                "RAG metadata validation issues: metadata_not_dict=%d, "
                "missing_hasKeywordMatch=%d, missing_isRecommendation=%d, "
                "invalid_hasKeywordMatch=%d, invalid_isRecommendation=%d, "
                "sample_results=%s",
                metadata_not_dict,
                missing_keyword_match,
                missing_recommendation,
                invalid_keyword_match,
                invalid_recommendation,
                sample_indices,
            )

        logger.info(
            f"[Filter] {len(results)} -> {len(filtered)} results "
            f"(min={self.min_vector_score}, safe_pass={self.fts_safe_pass_score})"
        )
        return filtered
