"""
WardInfoTool - Ward 정보 조회 도구

역할: 어르신의 위치 및 컨텍스트 정보를 조회하는 도구
- 위치 정보 조회: 날씨 조회 시 사용
- 기본 정보 조회: 이름, 나이 등 어르신 정보
"""

import logging

from livekit.agents import RunContext, function_tool

from ..userdata import SessionUserdata

logger = logging.getLogger(__name__)


class WardInfoToolMixin:
    """Ward 정보 조회 도구를 제공하는 Mixin."""

    @function_tool
    async def get_ward_location(
        self,
        context: RunContext[SessionUserdata],
    ) -> str:
        """
        어르신의 위치 정보(위도, 경도)를 조회합니다.

        날씨 조회 시 이 도구를 먼저 호출하여 위치 정보를 확인하세요.
        반환된 위도, 경도를 날씨 API에 전달합니다.

        Returns:
            위치 정보 문자열 (예: "위도: 37.5665, 경도: 126.9780")
            또는 "위치 정보가 없습니다."
        """
        userdata = context.userdata
        latitude = getattr(userdata, "latitude", None)
        longitude = getattr(userdata, "longitude", None)

        if latitude is not None and longitude is not None:
            result = f"위도: {latitude}, 경도: {longitude}"
            logger.info(f"get_ward_location called: {result}")
            return result
        else:
            logger.warning("get_ward_location called but no location info available")
            return "위치 정보가 없습니다."

    @function_tool
    async def get_ward_info(
        self,
        context: RunContext[SessionUserdata],
    ) -> str:
        """
        어르신의 기본 정보(이름, 나이 등)를 조회합니다.

        어르신에 대한 정보가 필요할 때 이 도구를 사용하세요.
        이름, 나이, 건강 상태 등의 정보를 확인할 수 있습니다.

        Returns:
            어르신 정보 문자열 또는 "어르신 정보가 없습니다."
        """
        userdata = context.userdata
        ward_context = getattr(userdata, "ward_context", None)

        if ward_context:
            logger.info(f"get_ward_info called: context_len={len(ward_context)}")
            return ward_context
        else:
            logger.warning("get_ward_info called but no ward_context available")
            return "어르신 정보가 없습니다."
