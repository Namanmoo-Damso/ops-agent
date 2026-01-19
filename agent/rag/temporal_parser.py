"""
Temporal Query Parser - 시간 표현 감지 및 날짜 변환

한국어 상대 시간 표현을 감지하고 날짜 범위로 변환:
- "이틀 전" → 2일 전 날짜
- "어제" → 어제 날짜
- "지난주" → 지난주 월~일
- "한 달 전" → 30일 전 날짜

Usage:
    from agent.rag.temporal_parser import TemporalParser

    parser = TemporalParser()
    result = parser.parse("이틀 전에 뭐했어?")
    # result: TemporalParseResult(
    #     has_temporal=True,
    #     start_date=datetime(2026, 1, 16, 0, 0, 0),
    #     end_date=datetime(2026, 1, 16, 23, 59, 59),
    #     expression="이틀 전"
    # )
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from zoneinfo import ZoneInfo

# 기본 시간대: 한국
DEFAULT_TZ = ZoneInfo("Asia/Seoul")


@dataclass
class TemporalParseResult:
    """시간 표현 파싱 결과"""

    has_temporal: bool
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    expression: Optional[str] = None


class TemporalParser:
    """한국어 시간 표현 파서"""

    def __init__(self, timezone: ZoneInfo = DEFAULT_TZ):
        self.tz = timezone

        # 상대 시간 표현 패턴 (우선순위 순)
        self.patterns = [
            # N일 전/후
            (r"(\d+)\s*일\s*전", self._parse_days_ago),
            (r"(\d+)\s*일\s*후", self._parse_days_later),
            # 어제, 그저께, 그제
            (r"어제", lambda m, now: self._make_day_range(now - timedelta(days=1))),
            (r"그저께|그제|엊그제", lambda m, now: self._make_day_range(now - timedelta(days=2))),
            # 이틀 전, 사흘 전
            (r"이틀\s*전", lambda m, now: self._make_day_range(now - timedelta(days=2))),
            (r"사흘\s*전", lambda m, now: self._make_day_range(now - timedelta(days=3))),
            # 지난주
            (r"지난\s*주", self._parse_last_week),
            # 이번 주
            (r"이번\s*주", self._parse_this_week),
            # N주 전
            (r"(\d+)\s*주\s*전", self._parse_weeks_ago),
            # 한 달 전, 두 달 전
            (r"한\s*달\s*전", lambda m, now: self._make_day_range(now - timedelta(days=30))),
            (r"두\s*달\s*전", lambda m, now: self._make_day_range(now - timedelta(days=60))),
            # N개월 전
            (r"(\d+)\s*개월\s*전", self._parse_months_ago),
            # 오늘
            (r"오늘", lambda m, now: self._make_day_range(now)),
        ]

    def parse(self, query: str) -> TemporalParseResult:
        """쿼리에서 시간 표현 감지 및 파싱

        Args:
            query: 사용자 쿼리 문자열

        Returns:
            TemporalParseResult: 파싱 결과
        """
        if not query:
            return TemporalParseResult(has_temporal=False)

        now = datetime.now(self.tz)

        for pattern, handler in self.patterns:
            match = re.search(pattern, query)
            if match:
                start_date, end_date = handler(match, now)
                return TemporalParseResult(
                    has_temporal=True,
                    start_date=start_date,
                    end_date=end_date,
                    expression=match.group(0),
                )

        return TemporalParseResult(has_temporal=False)

    def _make_day_range(self, dt: datetime) -> tuple[datetime, datetime]:
        """특정 날짜의 시작~끝 범위 생성"""
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _parse_days_ago(self, match: re.Match, now: datetime) -> tuple[datetime, datetime]:
        """N일 전"""
        days = int(match.group(1))
        target = now - timedelta(days=days)
        return self._make_day_range(target)

    def _parse_days_later(self, match: re.Match, now: datetime) -> tuple[datetime, datetime]:
        """N일 후"""
        days = int(match.group(1))
        target = now + timedelta(days=days)
        return self._make_day_range(target)

    def _parse_weeks_ago(self, match: re.Match, now: datetime) -> tuple[datetime, datetime]:
        """N주 전 (해당 주 전체)"""
        weeks = int(match.group(1))
        # 해당 주의 월요일
        target_monday = now - timedelta(days=now.weekday() + (weeks * 7))
        target_sunday = target_monday + timedelta(days=6)

        start = target_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = target_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _parse_last_week(self, match: re.Match, now: datetime) -> tuple[datetime, datetime]:
        """지난주 (월~일)"""
        # 이번 주 월요일
        this_monday = now - timedelta(days=now.weekday())
        # 지난주 월요일
        last_monday = this_monday - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)

        start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _parse_this_week(self, match: re.Match, now: datetime) -> tuple[datetime, datetime]:
        """이번 주 (월~오늘)"""
        # 이번 주 월요일
        this_monday = now - timedelta(days=now.weekday())

        start = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _parse_months_ago(self, match: re.Match, now: datetime) -> tuple[datetime, datetime]:
        """N개월 전 (해당 월 전체가 아닌 약 30일 단위)"""
        months = int(match.group(1))
        target = now - timedelta(days=months * 30)
        return self._make_day_range(target)


# 싱글톤 인스턴스
_temporal_parser: Optional[TemporalParser] = None


def get_temporal_parser() -> TemporalParser:
    """시간 파서 싱글톤 반환"""
    global _temporal_parser
    if _temporal_parser is None:
        _temporal_parser = TemporalParser()
    return _temporal_parser
