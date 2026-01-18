"""
KMA MCP Server - Korea Meteorological Administration MCP Server

Provides weather tools via FastMCP.
Uses official KMA API Hub (apihub.kma.go.kr) for weather data.
"""

import asyncio
import httpx
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo
from fastmcp import FastMCP
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kma-mcp-server")

# Create FastMCP server
mcp = FastMCP("KMA Weather Server")


# Add health endpoint for Docker healthcheck
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for Docker healthcheck."""
    return JSONResponse({"status": "healthy", "service": "kma-mcp"})

# API Configuration
KMA_API_BASE = "https://apihub.kma.go.kr"
KMA_AUTH_KEY = os.getenv("KMA_AUTH_KEY", "")
KST = ZoneInfo("Asia/Seoul")


# KMA Grid conversion constants
# Based on the LC (Lambert Conformal Conic) projection used by KMA
RE = 6371.00877  # Earth radius (km)
GRID = 5.0  # Grid spacing (km)
SLAT1 = 30.0  # Standard latitude 1
SLAT2 = 60.0  # Standard latitude 2
OLON = 126.0  # Reference longitude
OLAT = 38.0  # Reference latitude
XO = 43  # Reference X coordinate
YO = 136  # Reference Y coordinate


def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    """
    Convert latitude/longitude to KMA grid coordinates (nx, ny).
    
    Uses Lambert Conformal Conic projection parameters from KMA.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
    
    Returns:
        Tuple of (nx, ny) grid coordinates
    """
    DEGRAD = math.pi / 180.0
    
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    
    nx = int(ra * math.sin(theta) + XO + 0.5)
    ny = int(ro - ra * math.cos(theta) + YO + 0.5)
    
    return nx, ny


def get_weather_category_kr(category: str, value: str) -> str:
    """Convert KMA weather category code to Korean description."""
    try:
        val = float(value) if value else 0
    except ValueError:
        val = 0
    
    if category == "PTY":  # 강수형태
        codes = {0: "없음", 1: "비", 2: "비/눈", 3: "눈", 4: "소나기", 5: "빗방울", 6: "빗방울/눈날림", 7: "눈날림"}
        return codes.get(int(val), "알 수 없음")
    elif category == "SKY":  # 하늘상태
        codes = {1: "맑음", 3: "구름많음", 4: "흐림"}
        return codes.get(int(val), "알 수 없음")
    elif category == "T1H" or category == "TMP":  # 기온
        return f"{val}°C"
    elif category == "RN1" or category == "PCP":  # 1시간 강수량
        if val == 0:
            return "없음"
        return f"{val}mm"
    elif category == "REH":  # 습도
        return f"{int(val)}%"
    elif category == "VEC":  # 풍향
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((val + 22.5) / 45) % 16
        return directions[idx]
    elif category == "WSD":  # 풍속
        return f"{val}m/s"
    elif category == "POP":  # 강수확률
        return f"{int(val)}%"
    elif category == "TMN":  # 최저기온
        return f"{val}°C"
    elif category == "TMX":  # 최고기온
        return f"{val}°C"
    return value


def get_base_time_for_ultra_srt() -> str:
    """Get the appropriate base_time for 초단기실황 API (updated every hour)."""
    now = datetime.now(KST)
    # 초단기실황 is available at every hour (XX00)
    # Use the previous hour if we're close to the start of the hour
    if now.minute < 10:
        # Use previous hour's data
        hour = now.hour - 1
        if hour < 0:
            hour = 23
    else:
        hour = now.hour
    return f"{hour:02d}00"


def get_base_time_for_vilage_fcst() -> str:
    """Get the appropriate base_time for 단기예보 API."""
    now = datetime.now(KST)
    # 단기예보 is published at 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300
    base_times = [2, 5, 8, 11, 14, 17, 20, 23]
    current_hour = now.hour

    # Find the most recent base time
    base_hour = 23  # Default to previous day's last forecast
    for bt in reversed(base_times):
        if current_hour >= bt + 1:  # Allow 1 hour for data to be available
            base_hour = bt
            break

    return f"{base_hour:02d}00"


def get_mid_forecast_reg_id(lat: float, lon: float) -> str:
    """
    Get the region ID for 중기예보 API based on latitude/longitude.

    KMA uses region codes for medium-term forecasts.
    This maps coordinates to the nearest forecast region.

    Region codes: https://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp
    """
    # Simplified mapping based on major regions
    # Format: (min_lat, max_lat, min_lon, max_lon, reg_id)
    regions = [
        # 서울/경기/인천
        (36.8, 38.0, 126.0, 127.5, "11B00000"),
        # 강원도 영서
        (37.0, 38.5, 127.5, 128.5, "11D10000"),
        # 강원도 영동
        (37.0, 38.5, 128.5, 129.5, "11D20000"),
        # 충청북도
        (36.0, 37.2, 127.0, 128.2, "11C10000"),
        # 충청남도
        (35.8, 37.0, 126.0, 127.2, "11C20000"),
        # 전라북도
        (35.3, 36.2, 126.3, 127.8, "11F10000"),
        # 전라남도
        (34.0, 35.5, 126.0, 127.8, "11F20000"),
        # 경상북도
        (35.5, 37.2, 128.2, 130.0, "11H10000"),
        # 경상남도
        (34.5, 35.8, 127.8, 129.5, "11H20000"),
        # 제주도
        (33.0, 34.0, 126.0, 127.0, "11G00000"),
    ]

    for min_lat, max_lat, min_lon, max_lon, reg_id in regions:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return reg_id

    # Default to Seoul/Gyeonggi if no match
    return "11B00000"


def get_mid_ta_reg_id(lat: float, lon: float) -> str:
    """
    Get the region ID for 중기기온 API based on latitude/longitude.

    These are city-level codes for temperature forecasts.
    """
    # Simplified mapping for major cities
    regions = [
        # 서울
        (37.4, 37.7, 126.8, 127.2, "11B10101"),
        # 인천
        (37.3, 37.6, 126.5, 126.8, "11B20201"),
        # 수원
        (37.2, 37.4, 126.9, 127.1, "11B20601"),
        # 대전
        (36.2, 36.5, 127.2, 127.5, "11C20401"),
        # 청주
        (36.5, 36.8, 127.3, 127.6, "11C10301"),
        # 광주
        (35.0, 35.3, 126.7, 127.0, "11F20501"),
        # 전주
        (35.7, 36.0, 127.0, 127.3, "11F10201"),
        # 대구
        (35.7, 36.0, 128.4, 128.8, "11H10701"),
        # 부산
        (35.0, 35.3, 128.9, 129.2, "11H20201"),
        # 울산
        (35.4, 35.7, 129.2, 129.5, "11H20101"),
        # 강릉
        (37.6, 37.9, 128.8, 129.1, "11D20501"),
        # 춘천
        (37.8, 38.1, 127.6, 127.9, "11D10301"),
        # 제주
        (33.3, 33.6, 126.4, 126.7, "11G00201"),
    ]

    for min_lat, max_lat, min_lon, max_lon, reg_id in regions:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return reg_id

    # Default to Seoul if no match
    return "11B10101"


@mcp.tool
async def get_current_weather(latitude: float, longitude: float) -> str:
    """
    현재 날씨 정보를 조회합니다 (기상청 초단기실황 API 사용).

    Args:
        latitude: 위도 (예: 37.5665 for Seoul)
        longitude: 경도 (예: 126.9780 for Seoul)

    Returns:
        현재 날씨 정보 (기온, 습도, 풍속, 강수 상태)
    """
    if not KMA_AUTH_KEY:
        return "오류: KMA API 인증키가 설정되지 않았습니다. KMA_AUTH_KEY 환경변수를 확인해주세요."
    
    try:
        # Convert lat/lon to grid coordinates
        nx, ny = latlon_to_grid(latitude, longitude)
        logger.info(f"Converted ({latitude}, {longitude}) to grid ({nx}, {ny})")
        
        # Get current date and base time
        now = datetime.now(KST)
        base_date = now.strftime("%Y%m%d")
        base_time = get_base_time_for_ultra_srt()
        
        # Build API URL for 초단기실황
        url = f"{KMA_API_BASE}/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtNcst"
        params = {
            "pageNo": 1,
            "numOfRows": 100,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
            "authKey": KMA_AUTH_KEY,
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Parse response
        body = data.get("response", {}).get("body", {})
        items = body.get("items", {}).get("item", [])
        
        if not items:
            return f"날씨 정보를 가져올 수 없습니다. (위치: {latitude}, {longitude})"
        
        # Extract weather data
        weather_data = {}
        for item in items:
            category = item.get("category")
            value = item.get("obsrValue")
            weather_data[category] = value
        
        # Format response
        temp = weather_data.get("T1H", "N/A")
        humidity = weather_data.get("REH", "N/A")
        wind_speed = weather_data.get("WSD", "N/A")
        wind_dir = weather_data.get("VEC", "N/A")
        rain = weather_data.get("RN1", "0")
        pty = weather_data.get("PTY", "0")
        
        precip_type = get_weather_category_kr("PTY", pty)
        wind_dir_kr = get_weather_category_kr("VEC", wind_dir) if wind_dir != "N/A" else "N/A"
        
        result = f"""현재 날씨 정보 (기상청 제공):
- 기온: {temp}°C
- 습도: {humidity}%
- 풍속: {wind_speed}m/s ({wind_dir_kr})
- 강수형태: {precip_type}"""
        
        if float(rain) > 0:
            result += f"\n- 1시간 강수량: {rain}mm"
        
        logger.info(f"Weather query for ({latitude}, {longitude}) grid ({nx}, {ny}): {temp}°C")
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching weather: {e}")
        return f"날씨 정보를 가져오는 중 오류가 발생했습니다: HTTP {e.response.status_code}"
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return f"날씨 정보를 가져오는 중 오류가 발생했습니다: {str(e)}"


@mcp.tool
async def get_weather_forecast(
    latitude: float,
    longitude: float,
    days: int = 3
) -> str:
    """
    날씨 예보를 조회합니다 (기상청 단기예보 API 사용).

    Args:
        latitude: 위도
        longitude: 경도
        days: 예보 일수 (1-3일, 기본값 3일)

    Returns:
        일별 날씨 예보 (최고/최저 기온, 날씨 상태, 강수 확률)
    """
    if not KMA_AUTH_KEY:
        return "오류: KMA API 인증키가 설정되지 않았습니다. KMA_AUTH_KEY 환경변수를 확인해주세요."
    
    try:
        days = min(max(days, 1), 3)  # KMA short-term forecast is up to 3 days
        
        # Convert lat/lon to grid coordinates
        nx, ny = latlon_to_grid(latitude, longitude)
        
        # Get base date and time
        now = datetime.now(KST)
        base_date = now.strftime("%Y%m%d")
        base_time = get_base_time_for_vilage_fcst()
        
        # Build API URL for 단기예보
        url = f"{KMA_API_BASE}/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
        params = {
            "pageNo": 1,
            "numOfRows": 1000,  # Get enough rows for multi-day forecast
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
            "authKey": KMA_AUTH_KEY,
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Parse response
        body = data.get("response", {}).get("body", {})
        items = body.get("items", {}).get("item", [])
        
        if not items:
            return f"날씨 예보를 가져올 수 없습니다. (위치: {latitude}, {longitude})"
        
        # Group by date
        daily_data = {}
        for item in items:
            fcst_date = item.get("fcstDate")
            category = item.get("category")
            value = item.get("fcstValue")
            fcst_time = item.get("fcstTime")
            
            if fcst_date not in daily_data:
                daily_data[fcst_date] = {}
            
            # Store important values
            if category in ["TMN", "TMX", "POP", "SKY", "PTY"]:
                if category == "POP":
                    # Get max precipitation probability
                    existing = daily_data[fcst_date].get(category, "0")
                    if int(value) > int(existing):
                        daily_data[fcst_date][category] = value
                elif category in ["TMN", "TMX"]:
                    daily_data[fcst_date][category] = value
                elif category == "SKY" and fcst_time == "1200":  # Noon sky condition
                    daily_data[fcst_date][category] = value
                elif category == "PTY" and fcst_time == "1200":
                    daily_data[fcst_date][category] = value
        
        # Format response
        forecast_lines = [f"향후 {days}일 날씨 예보 (기상청 제공):"]
        
        sorted_dates = sorted(daily_data.keys())[:days]
        for date_str in sorted_dates:
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][dt.weekday()]
                formatted_date = f"{dt.month}월 {dt.day}일 ({weekday_kr})"
            except ValueError:
                formatted_date = date_str
            
            day_info = daily_data[date_str]
            tmn = day_info.get("TMN", "?")
            tmx = day_info.get("TMX", "?")
            pop = day_info.get("POP", "0")
            sky = day_info.get("SKY", "1")
            pty = day_info.get("PTY", "0")
            
            # Determine weather description
            if pty and int(pty) > 0:
                weather = get_weather_category_kr("PTY", pty)
            else:
                weather = get_weather_category_kr("SKY", sky)
            
            line = f"\n{formatted_date}\n   날씨: {weather}\n   기온: {tmn}°C ~ {tmx}°C"
            if int(pop) > 0:
                line += f"\n   강수 확률: {pop}%"
            
            forecast_lines.append(line)
        
        logger.info(f"Forecast query for ({latitude}, {longitude}): {days} days")
        return "\n".join(forecast_lines)

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching forecast: {e}")
        return f"날씨 예보를 가져오는 중 오류가 발생했습니다: HTTP {e.response.status_code}"
    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        return f"날씨 예보를 가져오는 중 오류가 발생했습니다: {str(e)}"


@mcp.tool
async def get_weekly_weather_forecast(
    latitude: float,
    longitude: float,
) -> str:
    """
    주간 날씨 예보를 조회합니다 (기상청 중기예보 API 사용).

    단기예보(3일)와 중기예보(4-7일)를 결합하여 7일간의 예보를 제공합니다.

    Args:
        latitude: 위도
        longitude: 경도

    Returns:
        7일간의 날씨 예보 (최고/최저 기온, 날씨 상태, 강수 확률)
    """
    if not KMA_AUTH_KEY:
        return "오류: KMA API 인증키가 설정되지 않았습니다. KMA_AUTH_KEY 환경변수를 확인해주세요."

    try:
        now = datetime.now(KST)
        forecast_lines = ["주간 날씨 예보 (기상청 제공):"]

        # Part 1: Get short-term forecast (days 1-3) using 단기예보
        nx, ny = latlon_to_grid(latitude, longitude)
        base_date = now.strftime("%Y%m%d")
        base_time = get_base_time_for_vilage_fcst()

        url_short = f"{KMA_API_BASE}/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
        params_short = {
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
            "authKey": KMA_AUTH_KEY,
        }

        short_term_data = {}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url_short, params=params_short)
            response.raise_for_status()
            data = response.json()

            body = data.get("response", {}).get("body", {})
            items = body.get("items", {}).get("item", [])

            for item in items:
                fcst_date = item.get("fcstDate")
                category = item.get("category")
                value = item.get("fcstValue")
                fcst_time = item.get("fcstTime")

                if fcst_date not in short_term_data:
                    short_term_data[fcst_date] = {}

                if category in ["TMN", "TMX", "POP", "SKY", "PTY"]:
                    if category == "POP":
                        existing = short_term_data[fcst_date].get(category, "0")
                        if int(value) > int(existing):
                            short_term_data[fcst_date][category] = value
                    elif category in ["TMN", "TMX"]:
                        short_term_data[fcst_date][category] = value
                    elif category == "SKY" and fcst_time == "1200":
                        short_term_data[fcst_date][category] = value
                    elif category == "PTY" and fcst_time == "1200":
                        short_term_data[fcst_date][category] = value

        # Part 2: Get medium-term forecast (days 4-7) using 중기예보
        # 중기예보 is published at 06:00 and 18:00
        mid_base_time = "0600" if now.hour < 18 else "1800"
        if now.hour < 6:
            # Use yesterday's 18:00 forecast
            mid_base_date = (now - timedelta(days=1)).strftime("%Y%m%d")
            mid_base_time = "1800"
        else:
            mid_base_date = base_date

        reg_id_land = get_mid_forecast_reg_id(latitude, longitude)
        reg_id_ta = get_mid_ta_reg_id(latitude, longitude)

        # Fetch 중기육상예보 (weather conditions)
        url_mid_land = f"{KMA_API_BASE}/api/typ02/openApi/MidFcstInfoService/getMidLandFcst"
        params_mid_land = {
            "pageNo": 1,
            "numOfRows": 10,
            "dataType": "JSON",
            "regId": reg_id_land,
            "tmFc": f"{mid_base_date}{mid_base_time}",
            "authKey": KMA_AUTH_KEY,
        }

        # Fetch 중기기온예보 (temperatures)
        url_mid_ta = f"{KMA_API_BASE}/api/typ02/openApi/MidFcstInfoService/getMidTa"
        params_mid_ta = {
            "pageNo": 1,
            "numOfRows": 10,
            "dataType": "JSON",
            "regId": reg_id_ta,
            "tmFc": f"{mid_base_date}{mid_base_time}",
            "authKey": KMA_AUTH_KEY,
        }

        mid_land_data = {}
        mid_ta_data = {}

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Fetch both in parallel
            land_response, ta_response = await asyncio.gather(
                client.get(url_mid_land, params=params_mid_land),
                client.get(url_mid_ta, params=params_mid_ta),
                return_exceptions=True,
            )

            # Parse 중기육상예보 (weather + precipitation)
            if isinstance(land_response, httpx.Response) and land_response.status_code == 200:
                land_body = land_response.json().get("response", {}).get("body", {})
                land_items = land_body.get("items", {}).get("item", [])
                if land_items:
                    item = land_items[0]
                    # Days 3-7: rnSt3Am, rnSt3Pm, wf3Am, wf3Pm, etc.
                    for day in range(3, 8):
                        mid_land_data[day] = {
                            "rain_am": item.get(f"rnSt{day}Am", 0),
                            "rain_pm": item.get(f"rnSt{day}Pm", 0),
                            "wf_am": item.get(f"wf{day}Am", ""),
                            "wf_pm": item.get(f"wf{day}Pm", ""),
                        }

            # Parse 중기기온예보 (min/max temps)
            if isinstance(ta_response, httpx.Response) and ta_response.status_code == 200:
                ta_body = ta_response.json().get("response", {}).get("body", {})
                ta_items = ta_body.get("items", {}).get("item", [])
                if ta_items:
                    item = ta_items[0]
                    # Days 3-7: taMin3, taMax3, etc.
                    for day in range(3, 8):
                        mid_ta_data[day] = {
                            "min": item.get(f"taMin{day}", "?"),
                            "max": item.get(f"taMax{day}", "?"),
                        }

        # Combine short-term and medium-term forecasts
        all_forecasts = []

        # Add short-term forecasts (days 1-3)
        sorted_short_dates = sorted(short_term_data.keys())[:3]
        for date_str in sorted_short_dates:
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][dt.weekday()]
                formatted_date = f"{dt.month}월 {dt.day}일 ({weekday_kr})"
            except ValueError:
                formatted_date = date_str

            day_info = short_term_data[date_str]
            tmn = day_info.get("TMN", "?")
            tmx = day_info.get("TMX", "?")
            pop = day_info.get("POP", "0")
            sky = day_info.get("SKY", "1")
            pty = day_info.get("PTY", "0")

            if pty and int(pty) > 0:
                weather = get_weather_category_kr("PTY", pty)
            else:
                weather = get_weather_category_kr("SKY", sky)

            line = f"\n{formatted_date}\n   날씨: {weather}\n   기온: {tmn}°C ~ {tmx}°C"
            if int(pop) > 0:
                line += f"\n   강수 확률: {pop}%"
            all_forecasts.append(line)

        # Add medium-term forecasts (days 4-7)
        for day_offset in range(4, 8):
            # Calculate date
            target_date = now + timedelta(days=day_offset - 1)
            weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][target_date.weekday()]
            formatted_date = f"{target_date.month}월 {target_date.day}일 ({weekday_kr})"

            # Get data from medium-term forecast (indexed by days from base)
            land_info = mid_land_data.get(day_offset, {})
            ta_info = mid_ta_data.get(day_offset, {})

            tmn = ta_info.get("min", "?")
            tmx = ta_info.get("max", "?")

            # Use PM weather as representative
            weather = land_info.get("wf_pm", "")
            if not weather:
                weather = land_info.get("wf_am", "맑음")

            # Average rain probability
            rain_am = land_info.get("rain_am", 0)
            rain_pm = land_info.get("rain_pm", 0)
            try:
                pop = max(int(rain_am), int(rain_pm))
            except (ValueError, TypeError):
                pop = 0

            line = f"\n{formatted_date}\n   날씨: {weather}\n   기온: {tmn}°C ~ {tmx}°C"
            if pop > 0:
                line += f"\n   강수 확률: {pop}%"
            all_forecasts.append(line)

        forecast_lines.extend(all_forecasts[:7])  # Ensure max 7 days

        logger.info(f"Weekly forecast query for ({latitude}, {longitude}): 7 days")
        return "\n".join(forecast_lines)

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching weekly forecast: {e}")
        return f"주간 날씨 예보를 가져오는 중 오류가 발생했습니다: HTTP {e.response.status_code}"
    except Exception as e:
        logger.error(f"Error fetching weekly forecast: {e}")
        return f"주간 날씨 예보를 가져오는 중 오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    print("=" * 50)
    print("KMA MCP Server - Korea Weather")
    print("Starting HTTP server on port 8002...")
    print("=" * 50)

    # Run with SSE transport for LiveKit compatibility
    mcp.run(transport="sse", host="0.0.0.0", port=8002)
