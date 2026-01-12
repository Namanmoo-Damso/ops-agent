"""
Korean Voice AI Agent for Elderly Care.

LiveKit Agents 1.3.10 - AgentServer pattern
"""
import asyncio
import os
import sys
import logging
import time

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    room_io,
)
from livekit.agents.metrics import STTMetrics, LLMMetrics, TTSMetrics
from livekit.plugins import aws, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from config import validate_env_vars, get_optional_config, ConfigError
from constants import TIMEOUT_RAG_CONTEXT_WARMUP
from personality.elderly_companion import ElderlyCompanionAgent, CallDirection
from userdata import SessionUserdata
from services.redis_pubsub import get_redis_client, publish_call_end
from services.api_client import fetch_call_context, notify_call_end
from handlers.transcript import TranscriptHandler
from handlers.takeover import TakeoverHandler
from handlers.session import (
    extract_ward_id,
    get_session_metadata,
    get_target_identity,
    SessionEndHandler,
    wait_for_participant,
)
from rag_client import get_shared_rag_client

# Agent name for routing
AGENT_NAME = os.getenv("AGENT_NAME", "voice-agent")

# Skip validation for download-files command (used during Docker build)
_is_download_command = "download-files" in sys.argv

if not _is_download_command:
    # Validate environment variables
    try:
        env_config = validate_env_vars()
        optional_config = get_optional_config()
    except ConfigError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
else:
    env_config = {}
    optional_config = {"LOG_LEVEL": "INFO"}

# Set logging
log_level = getattr(logging, optional_config["LOG_LEVEL"].upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create AgentServer instance
server = AgentServer()


def _create_task_logged(coro, name: str):
    """
    Create a background task and log exceptions to avoid silent failures.
    """
    task = asyncio.create_task(coro, name=name)

    def _log_error(t: asyncio.Task):
        try:
            exc = t.exception()
            if exc:
                logger.error(f"Background task '{name}' failed: {exc}", exc_info=True)
        except asyncio.CancelledError:
            pass

    task.add_done_callback(_log_error)
    return task


def prewarm(proc: JobProcess):
    """Prewarm function - loads models shared across sessions."""
    try:
        logger.info("Prewarming: Loading VAD model...")
        proc.userdata["vad"] = silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=0.5,
            activation_threshold=0.7,
        )
        logger.info("Prewarm complete")
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        raise RuntimeError("Prewarm failed - cannot start worker") from e


server.setup_fnc = prewarm


async def load_conversation_context(ward_id: str, timeout: float = TIMEOUT_RAG_CONTEXT_WARMUP) -> str:
    """
    Load recent conversation context for the ward using RAG.

    This provides the AI agent with memory of past conversations.
    Uses short timeout to avoid blocking session start.

    Args:
        ward_id: Ward UUID
        timeout: Maximum time to wait for RAG response (default: 2.0s)

    Returns:
        Formatted context string or empty string on failure/timeout
    """
    try:
        rag_client = get_shared_rag_client(timeout=timeout)

        # Get recent conversation history with timeout protection
        recent_context = await asyncio.wait_for(
            rag_client.get_recent_context(
                ward_id=ward_id,
                limit=5,  # Last 5 conversation chunks
            ),
            timeout=timeout,
        )

        if not recent_context:
            logger.info(f"No previous conversation history for ward: {ward_id}")
            return ""

        # Format context for AI - use .get() for safe access
        context_parts = ["=== 최근 대화 기록 ==="]
        for ctx in recent_context:
            # Safe access: if keys missing, skip this entry
            created_at = ctx.get("createdAt", "")
            text = ctx.get("text", "")
            if text:
                # Limit text length for context
                text_preview = text[:200] + "..." if len(text) > 200 else text
                context_parts.append(f"\n[{created_at}]\n{text_preview}")

        context_str = "\n".join(context_parts)
        logger.info(f"Loaded conversation context for ward {ward_id}: {len(recent_context)} chunks")

        return context_str

    except asyncio.TimeoutError:
        # Timeout is expected - don't block session start
        logger.warning(f"RAG context load timed out after {timeout}s for ward {ward_id}, continuing without context")
        return ""
    except Exception as e:
        # Any other error - continue without context
        logger.warning(f"Failed to load conversation context for ward {ward_id}: {e}")
        return ""


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

    # 🚀 START PRELOAD IMMEDIATELY - Parallel with WebRTC handshake
    # This loads weekly vectors from PGVector → Redis cache BEFORE agent joins
    # By the time first greeting happens, cache is ready (fast searches)
    if ward_id:
        logger.info(f"🚀 Starting weekly context preload in background for ward: {ward_id}")
        rag_client = get_shared_rag_client()
        # Convert CallDirection enum to string for API
        direction_str = "outbound" if call_direction == CallDirection.OUTBOUND else "inbound"
        _create_task_logged(
            rag_client.preload_weekly_context(ward_id, direction_str),
            name="preload_weekly_context",
        )

    # Wait for user participant to join
    logger.info("Waiting for user participant to join...")
    user_participant_identity = await wait_for_participant(ctx, timeout=30.0)

    if user_participant_identity:
        logger.info(f"✅ User participant joined: {user_participant_identity}")
    else:
        logger.warning("⚠️  User participant did not join within timeout")

    # Load conversation context in background with timeout
    # IMPORTANT: Don't block session start - RAG is optional enhancement
    # If RAG is slow/unavailable, agent still starts and greets user
    ward_context = ""
    try:
        # Try to load context with short timeout to avoid delaying session start
        ward_context = await load_conversation_context(ward_id, timeout=TIMEOUT_RAG_CONTEXT_WARMUP)
    except Exception as e:
        # If context loading fails for any reason, continue without it
        logger.warning(f"Context loading failed, starting agent without context: {e}")

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
        turn_detection=MultilingualModel(),
        min_endpointing_delay=0.3,
        max_endpointing_delay=2.0,
    )

    # Initialize handlers
    takeover_handler = TakeoverHandler(ctx.room, session)
    transcript_handler = TranscriptHandler(call_id, ctx.room)
    session_end_handler = SessionEndHandler(call_id, ward_id)
    
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

    # Pipeline timing metrics
    pipeline_logger = logging.getLogger("pipeline.metrics")
    pipeline_times = {}

    @session.on("user_input_transcribed")
    def on_transcribed(ev):
        pipeline_times["stt_end"] = time.time()

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        m = ev.metrics
        if isinstance(m, LLMMetrics):
            pipeline_times["llm"] = m.duration
            pipeline_logger.info(f"[LLM] ttft={m.ttft:.3f}s total={m.duration:.3f}s tokens={m.completion_tokens}")
        elif isinstance(m, TTSMetrics):
            pipeline_times["tts"] = m.duration
            pipeline_logger.info(f"[TTS] ttfb={m.ttfb:.3f}s total={m.duration:.3f}s")
            # Calculate total pipeline and derive STT time
            total = time.time() - pipeline_times.get("stt_end", time.time())
            llm_time = pipeline_times.get("llm", 0)
            tts_time = pipeline_times.get("tts", 0)
            stt_time = max(0, total - llm_time - tts_time)
            pipeline_logger.info(f"[PIPELINE] stt={stt_time:.3f}s llm={llm_time:.3f}s tts={tts_time:.3f}s total={total:.3f}s")

    # Register takeover handlers
    takeover_handler.register_event_handlers()

    # Wait for iOS participant (non-admin)
    target_identity = await wait_for_participant(ctx)
    logger.info(f"Target participant: {target_identity}")

    # Create and start agent with conversation context
    agent = ElderlyCompanionAgent(
        ward_context=ward_context,  # Empty string if RAG failed/timed out
        call_direction=call_direction
    )

    # Start takeover polling in background
    takeover_task = _create_task_logged(
        takeover_handler.start_polling(),
        name="takeover_polling",
    )

    # Start session (this blocks until session ends)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity=target_identity,
            close_on_disconnect=False,
        ),
    )

    # Session ended - cancel takeover polling
    takeover_task.cancel()
    try:
        await takeover_task
    except asyncio.CancelledError:
        pass

    # Run post-session tasks
    logger.info(f"Session ended for call: {call_id}")

    # Make sure pending transcript publications finish before call end
    logger.info("Waiting for transcript publications to complete before publishing call_end")
    await transcript_handler.wait_for_pending_publications(timeout=10.0)

    try:
        success = await publish_call_end(call_id, ward_id)
        if not success:
            await notify_call_end(call_id, ward_id)
        logger.info("Call end event published")
    except Exception as e:
        logger.error(f"Failed to publish call_end event: {e}")


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
