"""
ContextFormatter - RAG 검색 결과 포맷팅

역할: 검색된 Row 데이터를 LLM이 이해하기 좋은 형식으로 변환
- chunk_header에서 날짜 추출
- 상대 시간 정보 추가 (예: "약 3주 전")
- 마크다운 형식으로 구조화
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ContextFormatter:
    """RAG 검색 결과를 LLM 친화적 포맷으로 변환."""

    def __init__(self, timezone_name: str = "Asia/Seoul"):
        """
        Args:
            timezone_name: 시간대 이름
        """
        try:
            from zoneinfo import ZoneInfo
            self.tz = ZoneInfo(timezone_name)
        except ImportError:
            self.tz = timezone.utc

    def format_for_llm(
        self,
        results: list[dict[str, Any]],
        current_time: Optional[datetime] = None,
    ) -> str:
        """
        검색 결과를 LLM이 이해하기 좋은 형식으로 변환.

        Args:
            results: 필터링된 검색 결과
            current_time: 현재 시간 (기본: now)

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        if not results:
            return ""

        now = current_time or datetime.now(self.tz)
        formatted_parts = ["=== 어르신의 과거 기억 (실제 대화 기록) ===\n"]

        for i, result in enumerate(results, 1):
            formatted = self._format_single_result(result, now, i)
            if formatted:
                formatted_parts.append(formatted)

        if len(formatted_parts) == 1:
            return ""

        formatted_parts.append(
            "\n💡 위 대화 기록을 바탕으로 어르신의 질문에 답변하세요."
        )

        context = "\n".join(formatted_parts)
        logger.info(f"[Formatter] Formatted {len(results)} results ({len(context)} chars)")
        return context

    def _format_single_result(
        self,
        result: dict[str, Any],
        now: datetime,
        index: int,
    ) -> str:
        """단일 결과 포맷팅."""
        metadata = result.get("metadata", {})
        chunk_header = metadata.get("chunk_header", "")
        
        # chunk_header에서 날짜와 키워드 추출
        date_str, keywords = self._parse_chunk_header(chunk_header)
        relative_time = self._get_relative_time(date_str, now) if date_str else ""
        
        # 텍스트 추출
        text = (
            result.get("snippet") 
            or result.get("parentText") 
            or result.get("text", "")
        )
        
        if not text:
            return ""

        # 포맷 구성
        header_line = f"📅 {date_str}" if date_str else f"기억 #{index}"
        if keywords:
            header_line += f" ({keywords})"
        if relative_time:
            header_line += f" - {relative_time}"

        return f"""
{header_line}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{text.strip()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    def _parse_chunk_header(self, header: str) -> tuple[str, str]:
        """
        chunk_header에서 날짜와 키워드 추출.
        
        형식: [2026-01-10 | 건강관리 | 무릎통증, 정형외과]
        """
        if not header:
            return "", ""
        
        # [날짜 | 주제 | 키워드] 형식 파싱
        match = re.match(r"\[(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\]", header)
        if match:
            date_str = match.group(1)
            keywords = match.group(3).strip()
            return date_str, keywords
        
        # 날짜만 추출 시도
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", header)
        if date_match:
            return date_match.group(1), ""
        
        return "", ""

    def _get_relative_time(self, date_str: str, now: datetime) -> str:
        """날짜를 상대 시간으로 변환."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt.replace(tzinfo=self.tz)
            now_local = now.astimezone(self.tz) if now.tzinfo else now.replace(tzinfo=self.tz)
            
            diff = now_local - dt
            days = diff.days
            
            if days < 0:
                return "오늘"
            if days == 0:
                return "오늘"
            if days == 1:
                return "어제"
            if days < 7:
                return f"{days}일 전"
            if days < 14:
                return "지난주"
            if days < 30:
                weeks = days // 7
                return f"약 {weeks}주 전"
            if days < 60:
                return "한 달 전"
            
            months = days // 30
            return f"약 {months}개월 전"
        except Exception as error:
            logger.debug(
                "Failed to parse relative time for date=%s: %s",
                date_str,
                error,
            )
            return ""
