"""
Korean Voice AI Agent for Elderly Care.

LiveKit Agents 1.3.10 - AgentServer pattern
"""
import asyncio
import os
import sys
import logging

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    cli,
)
from livekit.plugins import aws, silero

from config import validate_env_vars, get_optional_config, ConfigError
from personality.elderly_companion import ElderlyCompanionAgent, CallDirection
from userdata import SessionUserdata
from services.redis_pubsub import get_redis_client
from services.api_client import fetch_call_context
from handlers.transcript import TranscriptHandler
from handlers.takeover import TakeoverHandler
from handlers.session import (
    extract_ward_id,
    get_session_metadata,
    get_target_identity,
    SessionEndHandler,
)

# Agent name for routing
AGENT_NAME = os.getenv("AGENT_NAME", "voice-agent")

# Validate environment variables
try:
    env_config = validate_env_vars()
    optional_config = get_optional_config()
except ConfigError as e:
    print(f"Configuration Error: {e}", file=sys.stderr)
    sys.exit(1)

# Set logging
log_level = getattr(logging, optional_config["LOG_LEVEL"].upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create AgentServer instance
server = AgentServer()


def prewarm(proc: JobProcess):
    """Prewarm function - loads models shared across sessions."""
    try:
        logger.info("Prewarming: Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=1.0,
            activation_threshold=0.7,
        )
        logger.info("Prewarm complete")
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        raise RuntimeError("Prewarm failed - cannot start worker") from e


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    """Main entrypoint for the AI agent when joining a room."""
    logger.info(f"Agent starting in room: {ctx.room.name}")

    # Initialize Redis and connect to room
    await asyncio.gather(get_redis_client(), ctx.connect())

    # Extract session metadata
    metadata = get_session_metadata(ctx)
    if metadata:
        logger.info(f"Session metadata: {metadata}")

    room_name = ctx.room.name
    call_id = metadata.get("callId")
    ward_id = metadata.get("wardId")

    # Fetch call context if needed
    if not call_id and room_name:
        context = await fetch_call_context(room_name)
        if context:
            call_id = context.get("callId") or call_id
            ward_id = ward_id or context.get("wardId")

    call_id = call_id or room_name
    ward_id = ward_id or extract_ward_id(ctx.room)

    # Determine call direction
    is_outbound = ctx.room.name.startswith('bot-')
    call_direction = CallDirection.OUTBOUND if is_outbound else CallDirection.INBOUND

    logger.info(f"Session info: ward_id={ward_id}, call_id={call_id}, direction={call_direction}")

    # Create session userdata
    userdata = SessionUserdata(
        ward_id=ward_id,
        call_id=call_id,
        call_direction=call_direction,
    )

    # Create agent session
    session = AgentSession[SessionUserdata](
        userdata=userdata,
        stt=aws.STT(language="ko-KR"),
        llm=aws.LLM(
            model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            temperature=0.7,
        ),
        tts=aws.TTS(voice="Seoyeon"),
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=1.0,
        max_endpointing_delay=5.0,
    )

    # Initialize handlers
    transcript_handler = TranscriptHandler(call_id, ctx.room, userdata)
    takeover_handler = TakeoverHandler(ctx.room, session)
    session_end_handler = SessionEndHandler(call_id, ward_id, userdata)

    # Register session event handlers
    @session.on("user_input_transcribed")
    def on_user_transcript(ev):
        transcript_handler.handle_user_transcript(ev, takeover_handler.takeover_active)

    @session.on("agent_speech_committed")
    def on_agent_speech(ev):
        transcript_handler.handle_agent_speech(ev)

    @session.on("conversation_item_added")
    def on_conversation_item(ev):
        transcript_handler.handle_conversation_item(ev)

    @session.on("session_end")
    def on_session_end(report):
        session_end_handler.handle_session_end(report)

    # Register takeover handlers
    takeover_handler.register_event_handlers()

    # Wait for participant
    await asyncio.sleep(3)
    target_identity = get_target_identity(ctx)

    # Create and start agent
    agent = ElderlyCompanionAgent(call_direction=call_direction)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            close_on_disconnect=False,
            participant_identity=target_identity,
        ),
    )

    # Start takeover polling
    await takeover_handler.start_polling()

    # Wait for session to end
    await session_end_handler.wait_for_completion()


if __name__ == "__main__":
    print("=" * 50)
    print("KOREAN VOICE ASSISTANT FOR ELDERLY CARE")
    print("LiveKit Agents 1.3.10 - AgentServer")
    print("=" * 50)
    try:
        cli.run_app(server)
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
