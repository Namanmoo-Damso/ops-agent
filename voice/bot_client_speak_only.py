"""Simple bot client that creates a bot room via the ops-api
and joins it as a LiveKit participant using the returned token.

Run this in the agents container or on a machine that can reach the
ops-api and LiveKit server.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentSession, RoomInputOptions, RoomOutputOptions
from livekit.plugins import aws, silero

from config import ConfigError, get_optional_config, validate_env_vars
from agents import ElderlyCompanionAgent
from userdata import SessionUserdata


# Load environment variables from ../.env if present
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


try:
    _env_config = validate_env_vars()
    _optional = get_optional_config()
except ConfigError as e:
    print(f"Configuration Error in bot_client: {e}")
    raise SystemExit(1)

log_level = getattr(_optional["LOG_LEVEL"].upper(), "INFO", "INFO")
logging.basicConfig(
    level=getattr(logging, _optional["LOG_LEVEL"].upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_bot_session() -> dict:
    """Call ops-api /v1/livekit/bot to create a bot room and token."""
    api_base = os.getenv("OPS_API_URL", "http://localhost:8080")
    admin_token = os.getenv("ADMIN_ACCESS_TOKEN")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{api_base}/v1/livekit/bot", json={}, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def run_bot() -> None:
    # Step 1: ask ops-api to create a bot session and dispatch the voice agent
    bot_session = await create_bot_session()

    livekit_url = bot_session["livekitUrl"]
    token = bot_session["token"]
    room_name = bot_session["roomName"]
    identity = bot_session["identity"]

    logger.info("Created bot session room=%s identity=%s", room_name, identity)

    # Step 2: connect to LiveKit as the bot-* participant
    room = rtc.Room()
    await room.connect(livekit_url, token)
    logger.info(
        "Bot joined LiveKit room '%s' as identity '%s'",
        room.name,
        room.local_participant.identity,
    )

    # Use TTS directly without AgentSession to avoid STT
    # Bot only speaks, doesn't listen
    tts = aws.TTS(voice="Seoyeon", sample_rate=24000)

    # Wait for agent to join
    await asyncio.sleep(2)

    logger.info(f"Room participants: {len(room.remote_participants)}")
    for p in room.remote_participants.values():
        logger.info(f"  Participant: {p.identity}")

    # Synthesize the greeting
    logger.info("Synthesizing bot greeting...")
    audio_stream = tts.synthesize("안녕하세요, 저는 봇입니다. 오늘 기분이 어떠신가요?")

    # Create audio source and track (24000 Hz, 1 channel to match TTS)
    audio_source = rtc.AudioSource(24000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot_voice", audio_source)

    # Publish the track
    logger.info("Publishing audio track...")
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    publication = await room.local_participant.publish_track(audio_track, options)
    logger.info(f"Published track: {publication.sid}")

    # Play the TTS audio through the track
    logger.info("Playing TTS audio...")
    async for synthesized_audio in audio_stream:
        await audio_source.capture_frame(synthesized_audio.frame)

    logger.info(f"Bot published tracks: {len(room.local_participant.track_publications)}")
    for track_sid, pub in room.local_participant.track_publications.items():
        logger.info(f"  Track: {pub.kind}, source={pub.source}, muted={pub.muted}")

    logger.info("Bot finished speaking, keeping connection alive...")

    # Keep the bot connected until interrupted
    try:
        await asyncio.Event().wait()
    finally:
        await room.disconnect()


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("Bot client interrupted, exiting...")


if __name__ == "__main__":
    main()
