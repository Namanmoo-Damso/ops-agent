"""
ContextFormatter - RAG 검색 결과 포맷팅

역할: 검색된 Row 데이터를 LLM이 이해하기 좋은 형식으로 변환
- chunk_header에서 날짜 추출
- 상대 시간 정보 추가 (예: "약 3주 전")
- 마크다운 형식으로 구조화
"""

import logging
import re
from datetime import datetime, timedelta, timezone
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
        formatted_parts = []

        for i, result in enumerate(results, 1):
            formatted = self._format_single_result(result, now, i)
            if formatted:
                formatted_parts.append(formatted)

        if not formatted_parts:
            return ""

        return "\n".join(formatted_parts)

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
            result.get("snippet") or result.get("parentText") or result.get("text", "")
        )

        if not text:
            return ""

        resolved_hints = []
        if date_str:
            resolved_hints = self._resolve_relative_dates(text, date_str, now)

        # 포맷 구성
        meta_parts = []
        if date_str:
            meta_parts.append(f"Date: {date_str}")

        if relative_time:
            meta_parts.append(f"({relative_time})")

        if keywords:
            meta_parts.append(f"Topic: {keywords}")

        # 깔끔한 헤더 구성
        if meta_parts:
            header = f"### {' '.join(meta_parts)}"
        else:
            header = f"### Memory #{index}"

        lines = [header]

        if resolved_hints:
            lines.append(f"> Hints: {', '.join(resolved_hints)}")

        lines.append(text.strip())
        lines.append("")  # Separator

        return "\n".join(lines)

    def _parse_chunk_header(self, header: str) -> tuple[str, str]:
        """
        chunk_header에서 날짜와 키워드 추출.

        형식: [2026-01-10 | 건강관리 | 무릎통증, 정형외과]
        """
        if not header:
            return "", ""

        # [날짜 | 주제 | 키워드] 형식 파싱
        match = re.match(
            r"\[(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\]", header
        )
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
            now_local = (
                now.astimezone(self.tz) if now.tzinfo else now.replace(tzinfo=self.tz)
            )

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

    def _resolve_relative_dates(
        self,
        text: str,
        date_str: str,
        now: datetime,
    ) -> list[str]:
        """상대 날짜 표현을 코드 레벨에서 해석해 힌트로 제공."""
        if not text or not date_str:
            return []

        try:
            base_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return []

        now_local = (
            now.astimezone(self.tz) if now.tzinfo else now.replace(tzinfo=self.tz)
        )
        now_date = now_local.date()

        hints: list[str] = []
        seen: set[str] = set()

        relative_map = {
            "오늘": 0,
            "내일": 1,
            "모레": 2,
            "글피": 3,
            "어제": -1,
            "그저께": -2,
            "그제": -2,
            "엊그제": -2,
        }

        word_pattern = re.compile(r"(오늘|내일|모레|글피|어제|그저께|그제|엊그제)")
        for match in word_pattern.finditer(text):
            word = match.group(1)
            if word in seen:
                continue
            seen.add(word)
            target_date = base_date + timedelta(days=relative_map[word])
            relative_label = self._format_relative_label(target_date, now_date)
            hints.append(f"{word}={target_date.isoformat()}({relative_label})")

        week_pattern = re.compile(r"(지난|이번|다음)\s*주\s*(월|화|수|목|금|토|일)요일")
        day_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
        week_start = base_date - timedelta(days=base_date.weekday())

        for match in week_pattern.finditer(text):
            phrase = match.group(0)
            if phrase in seen:
                continue
            seen.add(phrase)
            prefix = match.group(1)
            day_name = match.group(2)
            day_index = day_map[day_name]

            if prefix == "지난":
                target_date = week_start - timedelta(days=7) + timedelta(days=day_index)
            elif prefix == "다음":
                target_date = week_start + timedelta(days=7) + timedelta(days=day_index)
            else:
                target_date = week_start + timedelta(days=day_index)

            relative_label = self._format_relative_label(target_date, now_date)
            hints.append(f"{phrase}={target_date.isoformat()}({relative_label})")

        return hints

    @staticmethod
    def _format_relative_label(target_date, now_date) -> str:
        """해석된 날짜를 현재 기준 상대 표현으로 변환."""
        diff_days = (target_date - now_date).days
        if diff_days == 0:
            return "오늘"
        if diff_days == 1:
            return "내일"
        if diff_days == 2:
            return "모레"
        if diff_days == 3:
            return "글피"
        if diff_days == -1:
            return "어제"
        if diff_days == -2:
            return "그저께"
        if diff_days < 0:
            return f"{abs(diff_days)}일 전"
        return f"{diff_days}일 후"
