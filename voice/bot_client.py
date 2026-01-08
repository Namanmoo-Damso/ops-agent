"""Bot client that creates a bot room via the ops-api and joins it as a
LiveKit participant. This bot can both SPEAK (TTS) and LISTEN (STT) to
have two-way conversations with the simple_agent.

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
from livekit.agents.stt import SpeechEventType
from livekit.plugins import aws, silero

from config import ConfigError, get_optional_config, validate_env_vars


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


class SimpleBotAgent:
    """Simple bot agent that responds to agent's speech."""

    def __init__(self):
        self.conversation_count = 0
        self.responses = [
            "네, 잘 지내고 있어요. 오늘 날씨가 좋네요.",
            "그렇군요. 요즘 건강은 어떠세요?",
            "네, 알겠습니다. 감사합니다.",
            "좋은 하루 보내세요!",
        ]

    def get_response(self, agent_said: str) -> str:
        """Get a simple response based on conversation count."""
        response = self.responses[self.conversation_count % len(self.responses)]
        self.conversation_count += 1
        logger.info(f"Bot responding with: {response}")
        return response


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

    # Initialize STT, TTS, and VAD for two-way communication
    stt = aws.STT(language="ko-KR")
    tts = aws.TTS(voice="Seoyeon", sample_rate=24000)
    vad = silero.VAD.load(
        min_speech_duration=0.3,
        min_silence_duration=1.0,
        activation_threshold=0.7,
    )

    bot_agent = SimpleBotAgent()

    # Create audio source and track for publishing bot's voice
    audio_source = rtc.AudioSource(24000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot_voice", audio_source)

    # Lock to prevent concurrent speaking
    speak_lock = asyncio.Lock()
    # Flag to indicate bot is speaking (pause STT processing)
    is_speaking = False

    # Helper function to speak via TTS
    async def speak(text: str):
        """Synthesize and play TTS audio."""
        nonlocal is_speaking
        async with speak_lock:
            is_speaking = True
            logger.info(f"Bot speaking: {text}")
            audio_stream = tts.synthesize(text)
            async for synthesized_audio in audio_stream:
                await audio_source.capture_frame(synthesized_audio.frame)
            logger.info("Bot finished speaking")
            # Wait a bit for audio to finish playing before resuming STT
            await asyncio.sleep(0.5)
            is_speaking = False

    async def process_audio_track(track: rtc.Track, participant: rtc.RemoteParticipant):
        """Process incoming audio track with STT."""
        logger.info(f"Starting to process audio from {participant.identity}")

        audio_stream = rtc.AudioStream(track)
        stt_stream = stt.stream()
        frame_count = 0

        # Task to read transcripts
        async def read_transcripts():
            logger.info("STT transcript reader started")
            try:
                async for event in stt_stream:
                    logger.info(f"STT event: type={event.type}, alternatives={len(event.alternatives) if event.alternatives else 0}")
                    if event.alternatives:
                        for alt in event.alternatives:
                            logger.info(f"  Alternative: '{alt.text}'")
                    # Check for final transcript
                    if event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                        transcript = event.alternatives[0].text
                        if transcript and transcript.strip():
                            logger.info(f"Agent said: {transcript}")
                            # Generate and speak response
                            response = bot_agent.get_response(transcript)
                            await speak(response)
            except Exception as e:
                logger.error(f"Error in transcript reader: {e}", exc_info=True)
            logger.info("STT transcript reader ended")

        transcript_task = asyncio.create_task(read_transcripts())

        # Feed audio to STT (skip frames while bot is speaking to avoid echo)
        try:
            async for frame_event in audio_stream:
                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100 frames
                    logger.info(f"Received {frame_count} audio frames from {participant.identity} (speaking={is_speaking})")
                # Only process audio when bot is NOT speaking
                if not is_speaking:
                    stt_stream.push_frame(frame_event.frame)
        except Exception as e:
            logger.error(f"Error reading audio stream: {e}")

        logger.info(f"Audio stream ended after {frame_count} frames")
        await stt_stream.aclose()
        await transcript_task

    # Track active audio processing tasks
    active_audio_tasks: dict[str, asyncio.Task] = {}

    # Track subscription handler - listen to agent's audio
    # IMPORTANT: Register BEFORE publishing our track so we catch all subscriptions
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"Track subscribed: kind={track.kind}, participant={participant.identity}, sid={publication.sid}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"=== AUDIO TRACK SUBSCRIBED from {participant.identity} ===")
            # Only process agent tracks, not our own echoes
            if participant.identity.startswith("agent-"):
                task_key = f"{participant.identity}_{publication.sid}"
                if task_key not in active_audio_tasks:
                    logger.info(f"Starting audio processing for {participant.identity}")
                    task = asyncio.create_task(process_audio_track(track, participant))
                    active_audio_tasks[task_key] = task
                else:
                    logger.info(f"Already processing audio from {participant.identity}")
            else:
                logger.info(f"Ignoring audio track from non-agent: {participant.identity}")

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"Track published: kind={publication.kind}, participant={participant.identity}, subscribed={publication.subscribed}")

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    # Publish the track immediately so agent can hear us
    logger.info("Publishing audio track...")
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    publication = await room.local_participant.publish_track(audio_track, options)
    logger.info(f"Published track: {publication.sid}")

    # Wait for agent to join
    await asyncio.sleep(2)

    logger.info(f"Room participants: {len(room.remote_participants)}")
    agent_identity = None
    for p in room.remote_participants.values():
        logger.info(f"  Participant: {p.identity}, tracks: {len(p.track_publications)}")
        if p.identity.startswith("agent-"):
            agent_identity = p.identity
            logger.info(f"Bot will listen to agent: {agent_identity}")
        # Check for existing audio tracks from this participant
        for sid, track_pub in p.track_publications.items():
            logger.info(f"    Track: sid={sid}, kind={track_pub.kind}, subscribed={track_pub.subscribed}, track={track_pub.track}")
            if track_pub.kind == rtc.TrackKind.KIND_AUDIO and p.identity.startswith("agent-"):
                if track_pub.track:
                    task_key = f"{p.identity}_{sid}"
                    if task_key not in active_audio_tasks:
                        logger.info(f"Found existing audio track from {p.identity}, starting processing...")
                        task = asyncio.create_task(process_audio_track(track_pub.track, p))
                        active_audio_tasks[task_key] = task
                else:
                    logger.info(f"Audio track from {p.identity} exists but not yet subscribed, waiting...")

    if not agent_identity:
        logger.warning("No agent found in room yet, will wait for agent to join...")

    logger.info(f"Bot published tracks: {len(room.local_participant.track_publications)}")
    for track_sid, pub in room.local_participant.track_publications.items():
        logger.info(f"  Track: {pub.kind}, source={pub.source}, muted={pub.muted}")

    logger.info("Bot ready and listening for agent speech...")

    # Send initial greeting to start the conversation
    await asyncio.sleep(1)  # Give agent time to subscribe to our track
    await speak("안녕하세요, 저는 테스트 봇입니다. 잘 들리시나요?")

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
