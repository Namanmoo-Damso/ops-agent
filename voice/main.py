import sys
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, JobRequest
from livekit import rtc
import logging

from config import validate_env_vars, get_optional_config, ConfigError

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Validate environment variables before proceeding
try:
    env_config = validate_env_vars()
    optional_config = get_optional_config()
except ConfigError as e:
    print(f"❌ Configuration Error: {e}", file=sys.stderr)
    sys.exit(1)

# Set logging
log_level = getattr(logging, optional_config["LOG_LEVEL"].upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 1. 방에 참여했을 때 수행할 동작 (가공 공장의 실제 로직)
async def entrypoint(ctx: JobContext):
    """
    Main entrypoint when agent joins a room.

    Args:
        ctx: JobContext from LiveKit agent framework
    """
    try:
        logger.info(f"Agent attempting to join room: {ctx.room.name}")

        await ctx.connect()
        # 접속 후 메타데이터를 설정해 프론트에서 필터링할 수 있게 합니다.
        await ctx.room.local_participant.set_metadata(
            '{"type": "agent", "name": "AI"}'
        )
        logger.info(f"Successfully connected to room: {ctx.room.name}")

        # 참여자가 트랙(오디오/비디오)을 보낼 때 감지
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Audio track subscribed from: {participant.identity} (ID: {participant.sid})")
                # 나중에 여기에 AI(STT) 로직이 들어갑니다.
    except Exception as e:
        logger.error(f"Failed to join or connect to room {ctx.room.name}: {e}")
        raise

# 2. 서버의 일거리 요청(Job Request)을 어떻게 처리할지 결정
async def request_fnc(req: JobRequest):
    """
    Handle incoming job requests from LiveKit server.

    Args:
        req: JobRequest from LiveKit
    """
    try:
        logger.info(f"Received job request for room: {req.room.name}")
        # 모든 요청을 수락합니다.
        # 실제 서비스에서는 여기서 특정 방만 수락하거나 권한을 확인할 수 있습니다.
        await req.accept(name="AI")
    except Exception as e:
        logger.error(f"Failed to accept job request for room {req.room.name}: {e}")
        raise

# 3. 워커 실행 설정
if __name__ == "__main__":
    print("=" * 50)
    print("AI AGENT WORKER (Basic)")
    print("=" * 50)
    try:
        cli.run_app(WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc, # ★ 이 부분을 추가해야 서버의 요청을 수락합니다.
        ))
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
