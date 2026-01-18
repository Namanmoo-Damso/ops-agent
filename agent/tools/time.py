"""
TimeTool - get_current_time 도구 정의

역할: 현재 시간을 반환하는 간단한 도구
- 시간 질문 시 실시간 시간 제공
- 한국 시간(KST) 기준
"""

import logging
from datetime import datetime

from livekit.agents import RunContext, function_tool

from ..constants import AGENT_TZINFO
from ..userdata import SessionUserdata

logger = logging.getLogger(__name__)


class TimeToolMixin:
    """현재 시간 조회 도구를 제공하는 Mixin."""

    @function_tool
    async def get_current_time(
        self,
        context: RunContext[SessionUserdata],
    ) -> str:
        """
        현재 시간을 알려줍니다.

        사용자가 '지금 몇 시야?', '현재 시간', '시간 알려줘' 등
        시간을 묻는 질문을 하면 이 도구를 사용하세요.

        Returns:
            현재 한국 시간 (예: "2026년 1월 19일 오전 10시 30분")
        """
        now = datetime.now(AGENT_TZINFO)

        # Format time naturally for Korean elderly
        hour = now.hour
        minute = now.minute

        # AM/PM in Korean
        if hour < 12:
            period = "오전"
            display_hour = hour if hour > 0 else 12
        else:
            period = "오후"
            display_hour = hour - 12 if hour > 12 else 12

        # Format the time string
        time_str = f"{now.year}년 {now.month}월 {now.day}일 {period} {display_hour}시"
        if minute > 0:
            time_str += f" {minute}분"

        logger.info(f"get_current_time called: {time_str}")

        return f"현재 시간: {time_str} (한국 시간)"
