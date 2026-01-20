"""
Dynamic Token Budget - 질문 유형에 따른 동적 max_tokens 결정

사용자 질문의 복잡도를 분석하여 적절한 응답 길이를 결정합니다.
- 단순 질문 (시간, 이름): 40~50 토큰
- 일반 대화: 80~100 토큰
- 상세 설명 필요 (건강, 조언): 120~150 토큰
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 토큰 범위 설정
MIN_TOKENS = 40
DEFAULT_TOKENS = 100
MAX_TOKENS = 150


@dataclass
class TokenBudget:
    """토큰 예산 결과"""

    max_tokens: int
    reason: str


# 짧은 응답 패턴 (40-50 토큰)
SHORT_PATTERNS = [
    r"몇\s*시",  # 몇 시야?
    r"지금\s*시간",
    r"오늘\s*며칠",
    r"무슨\s*요일",
    r"누구야",
    r"이름이?\s*뭐",
    r"네가\s*누구",
    r"응|어|네|아니",  # 단답
    r"^.{1,5}$",  # 5자 이하 짧은 입력
]

# 긴 응답 패턴 (120-150 토큰)
LONG_PATTERNS = [
    r"어떻게\s*(하|해|된)",
    r"왜\s*(그런|이런|그래)",
    r"설명해?\s*줘",
    r"알려\s*줘",
    r"자세히",
    r"건강|아프|아파|병원|약",
    r"걱정|불안|우울|힘들",
    r"조언|도움",
    r"방법|뭘\s*해야",
]

# 중간 응답 패턴 (80-100 토큰) - 기본값
MEDIUM_PATTERNS = [
    r"뭐\s*했",
    r"어디\s*갔",
    r"뭐\s*먹",
    r"날씨",
    r"오늘|어제|내일",
]


def _match_patterns(text: str, patterns: list[str]) -> Optional[str]:
    """패턴 매칭 및 매칭된 패턴 반환"""
    for pattern in patterns:
        if re.search(pattern, text):
            return pattern
    return None


def calculate_token_budget(user_message: str) -> TokenBudget:
    """
    사용자 메시지 분석하여 적절한 max_tokens 결정

    Args:
        user_message: 사용자 입력 텍스트

    Returns:
        TokenBudget: max_tokens와 결정 이유
    """
    if not user_message:
        return TokenBudget(max_tokens=DEFAULT_TOKENS, reason="empty_input")

    text = user_message.strip().lower()

    # 1. 짧은 응답 패턴 체크
    if matched := _match_patterns(text, SHORT_PATTERNS):
        tokens = MIN_TOKENS
        logger.debug(f"[TokenBudget] SHORT pattern matched: {matched} → {tokens}")
        return TokenBudget(max_tokens=tokens, reason=f"short:{matched}")

    # 2. 긴 응답 패턴 체크
    if matched := _match_patterns(text, LONG_PATTERNS):
        tokens = MAX_TOKENS
        logger.debug(f"[TokenBudget] LONG pattern matched: {matched} → {tokens}")
        return TokenBudget(max_tokens=tokens, reason=f"long:{matched}")

    # 3. 중간 패턴 체크
    if matched := _match_patterns(text, MEDIUM_PATTERNS):
        tokens = DEFAULT_TOKENS
        logger.debug(f"[TokenBudget] MEDIUM pattern matched: {matched} → {tokens}")
        return TokenBudget(max_tokens=tokens, reason=f"medium:{matched}")

    # 4. 길이 기반 휴리스틱
    msg_len = len(user_message)
    if msg_len < 10:
        tokens = MIN_TOKENS + 10  # 50
    elif msg_len < 30:
        tokens = DEFAULT_TOKENS  # 100
    else:
        # 긴 질문은 보통 긴 답변 필요
        tokens = min(DEFAULT_TOKENS + (msg_len - 30), MAX_TOKENS)

    logger.debug(f"[TokenBudget] Length-based: len={msg_len} → {tokens}")
    return TokenBudget(max_tokens=tokens, reason=f"length:{msg_len}")


def get_max_tokens(user_message: str) -> int:
    """간편 함수: max_tokens 값만 반환"""
    return calculate_token_budget(user_message).max_tokens
