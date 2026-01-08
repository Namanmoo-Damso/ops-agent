"""
Korean Voice AI Agent for Elderly Care.

LiveKit Agents 1.3.10 - AgentServer pattern
"""
import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from urllib.parse import quote
import json

import httpx
import redis
import redis.asyncio as redis_async
from dotenv import load_dotenv
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
from agents.elderly_companion import ElderlyCompanionAgent, CallDirection
from userdata import SessionUserdata

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Validate environment variables before proceeding
try:
    env_config = validate_env_vars()
    optional_config = get_optional_config()
except ConfigError as e:
    print(f"Configuration Error: {e}", file=sys.stderr)
    sys.exit(1)

# Set logging level from environment
log_level = getattr(logging, optional_config["LOG_LEVEL"].upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Redis client (async) initialization
redis_client = None
redis_init_lock = asyncio.Lock()


async def init_redis_client():
    """Initialize Redis client asynchronously to avoid blocking the event loop."""
    global redis_client
    if redis_client:
        return

    async with redis_init_lock:
        if redis_client:
            return

        try:
            redis_client = redis_async.from_url(
                env_config["REDIS_URL"],
                decode_responses=True,
            )
            await redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            redis_client = None
        except Exception as e:
            logger.error(f"An unexpected error occurred with Redis: {e}")
            redis_client = None


# API configuration
API_BASE = os.getenv("API_BASE_URL")
if not API_BASE:
    logger.warning("API_BASE_URL not set, using default localhost:3000")
    API_BASE = "http://localhost:3000"

API_INTERNAL_TOKEN = os.getenv("API_INTERNAL_TOKEN")

# Timeouts (seconds)
TIMEOUT_RAG_INDEXING = 5.0
TIMEOUT_CALL_ANALYSIS = 5.0
TIMEOUT_CALL_CONTEXT = 5.0
TIMEOUT_CALL_END = 5.0
TIMEOUT_POST_SESSION = 10.0

# Create AgentServer instance
server = AgentServer()


def prewarm(proc: JobProcess):
    """
    Prewarm function - runs once when the worker starts.

    Use this to load models that should be shared across sessions.
    """
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


# Register prewarm function
server.setup_fnc = prewarm


def extract_ward_id(room) -> str:
    """
    Extract ward ID from room information.

    Supported formats:
    - "call_{ward_id}_{timestamp}" -> returns ward_id
    - "room-{uuid}" or other -> returns room name as fallback
    """
    room_name = room.name
    if "_" in room_name:
        parts = room_name.split("_")
        if len(parts) >= 2 and parts[0] == "call":
            ward_id = parts[1]
            if ward_id and len(ward_id) > 0:
                return ward_id

    # Fallback: use room name as ward_id
    logger.info(f"Using room name as ward_id: {room_name}")
    return room_name


def _parse_metadata(raw) -> dict:
    """Parse metadata payload into a dict."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def get_session_metadata(ctx: JobContext) -> dict:
    """Get dispatch or room metadata for the current session."""
    job = getattr(ctx, "job", None)
    raw = getattr(job, "metadata", None) if job else None
    if not raw:
        raw = getattr(ctx.room, "metadata", None)
    return _parse_metadata(raw)


def _get_auth_headers() -> dict:
    """Get authentication headers for internal API calls."""
    headers = {"Content-Type": "application/json"}
    if API_INTERNAL_TOKEN:
        headers["Authorization"] = f"Bearer {API_INTERNAL_TOKEN}"
    return headers


async def trigger_rag_indexing(call_id: str, ward_id: str):
    """Trigger RAG indexing after session ends."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/v1/rag/index",
                json={
                    "callId": call_id,
                    "wardId": ward_id,
                },
                headers=_get_auth_headers(),
                timeout=TIMEOUT_RAG_INDEXING,
            )
            logger.info(f"RAG indexing triggered: call={call_id}")
    except Exception as e:
        logger.error(f"RAG indexing trigger failed: {e}")


async def trigger_call_end(call_id: str):
    """Inform API that the call ended so it can finalize state and summaries."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{API_BASE}/v1/calls/end",
                json={"callId": call_id},
                headers=_get_auth_headers(),
                timeout=TIMEOUT_CALL_END,
            )
            logger.info(f"Call end notified: call={call_id}")
    except Exception as e:
        logger.error(f"Call end trigger failed: {e}")


async def fetch_call_context(room_name: str) -> Optional[dict]:
    """Resolve call metadata from the backend using the LiveKit room name."""
    if not room_name:
        return None

    try:
        encoded_room = quote(room_name, safe='')
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_BASE}/v1/calls/room/{encoded_room}/context",
                headers=_get_auth_headers(),
                timeout=TIMEOUT_CALL_CONTEXT,
            )
            response.raise_for_status()
            context = response.json()
            logger.info(f"Resolved call context for room={room_name}: {context}")
            return context
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response else 'unknown'
        if status == 404:
            logger.warning(f"No call record found for room={room_name}")
        else:
            logger.error(
                f"Call context request failed room={room_name} status={status} error={exc}"
            )
    except Exception as exc:
        logger.error(f"Call context request error room={room_name}: {exc}")

    return None


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the AI agent when joining a room.

    Uses AgentServer decorator pattern (LiveKit 1.3+).
    """
    logger.info(f"Agent starting in room: {ctx.room.name}")
    await asyncio.gather(init_redis_client(), ctx.connect())

    metadata = get_session_metadata(ctx)
    if metadata:
        logger.info(f"Session metadata: {metadata}")

    # Extract identifiers
    room_name = ctx.room.name
    call_id = metadata.get("callId")
    ward_id = metadata.get("wardId")

    if not call_id and room_name:
        context = await fetch_call_context(room_name)
        if context:
            call_id = context.get("callId") or call_id
            if context.get("wardId"):
                ward_id = ward_id or context.get("wardId")

    call_id = call_id or room_name
    ward_id = ward_id or extract_ward_id(ctx.room)

    # Determine call direction based on room name
    # Rooms starting with "bot-" are outbound calls (agent initiates)
    # All other rooms are inbound calls (user initiates)
    is_outbound = ctx.room.name.startswith('bot-')
    call_direction = CallDirection.OUTBOUND if is_outbound else CallDirection.INBOUND

    logger.info(f"Session info: ward_id={ward_id}, call_id={call_id}, direction={call_direction}")

    # Create session userdata
    userdata = SessionUserdata(
        ward_id=ward_id,
        call_id=call_id,
        call_direction=call_direction,
    )

    # Create agent session with userdata
    session = AgentSession[SessionUserdata](
        userdata=userdata,
        stt=aws.STT(language="ko-KR"),
        llm=aws.LLM(
            model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
            temperature=0.7,
        ),
        tts=aws.TTS(voice="Seoyeon"),
        vad=ctx.proc.userdata["vad"],  # Use prewarmed VAD
        # Session timing settings
        min_endpointing_delay=1.0,
        max_endpointing_delay=5.0,
    )

    async def add_transcript_to_redis(speaker_type: str, text: str):
        """Helper to create and push transcript to Redis."""
        if not redis_client:
            logger.warning("Redis client not available, skipping transcript storage.")
            return

        try:
            timestamp = datetime.utcnow().isoformat()
            transcript_entry = {
                "speaker": speaker_type,
                "text": text,
                "timestamp": timestamp,
            }
            redis_key = f"call:{call_id}:transcripts"
            pipe = redis_client.pipeline()
            pipe.rpush(redis_key, json.dumps(transcript_entry, ensure_ascii=False))
            pipe.expire(redis_key, 3600 * 24)  # 24 hours
            await pipe.execute()
            logger.debug(f"Saved to Redis: {speaker_type} - {text}")
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to save transcript to Redis: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving to Redis: {e}")

    # Event: User transcript received
    @session.on("user_input_transcribed")
    def on_user_transcript(ev):
        """Capture user transcripts in real-time."""
        if ev.is_final:
            # Store in-memory
            userdata.add_transcript("user", ev.transcript)
            logger.debug(f"User transcript: {ev.transcript}")
            # Store in Redis
            asyncio.create_task(add_transcript_to_redis("user", ev.transcript))

    # Event: Agent speech
    @session.on("agent_speech_committed")
    def on_agent_speech(ev):
        """Capture agent responses."""
        if hasattr(ev, 'content') and ev.content:
            # Store in-memory
            userdata.add_transcript("agent", ev.content)
            logger.debug(f"Agent response: {ev.content}")
            # Store in Redis
            asyncio.create_task(add_transcript_to_redis("agent", ev.content))

    session_end_event = asyncio.Event()
    post_session_task = None
    # Helper to broadcast transcripts to frontend
    async def broadcast_transcript(role: str, text: str):
        """Broadcast transcript to room via data packet."""
        try:
            payload = json.dumps({
                "type": "transcript",
                "role": role,
                "text": text,
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            })
            await ctx.room.local_participant.publish_data(
                payload,
                reliable=True,
            )
            logger.info(f"📡 Broadcasted {role} transcript: {text[:20]}...")
        except Exception as e:
            logger.error(f"Failed to broadcast transcript: {e}")

    # Event: Conversation item added (captures both user and agent messages)
    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        """Capture agent responses when they are added to conversation."""
        try:
            item = ev.item if hasattr(ev, 'item') else ev
            # Check if this is an agent message
            if hasattr(item, 'role') and item.role == 'assistant':
                # Get the text content - handle various formats
                content = None
                if hasattr(item, 'text') and item.text:
                    content = item.text
                elif hasattr(item, 'content'):
                    # content can be a list of content parts or a string
                    raw_content = item.content
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        # Extract text from list items
                        text_parts = []
                        for part in raw_content:
                            if isinstance(part, str):
                                text_parts.append(part)
                            elif hasattr(part, 'text'):
                                text_parts.append(part.text)
                            elif hasattr(part, '__str__'):
                                text_parts.append(str(part))
                        content = ' '.join(text_parts)
                    else:
                        content = str(raw_content)
                
                if content and content.strip():
                    userdata.add_transcript("agent", content)
                    logger.info(f"🤖 Agent response: {content}")
                    # Broadcast to frontend
                    asyncio.create_task(broadcast_transcript("agent", content))
            else:
                logger.debug(f"Conversation item added: role={getattr(item, 'role', 'unknown')}")
        except Exception as e:
            logger.error(f"Error in conversation_item_added handler: {e}")

    # Event: Speech created (backup - logs when agent starts speaking)
    @session.on("speech_created")
    def on_speech_created(ev):
        """Log when agent speech is created."""
        logger.info(f"🎤 Speech created: source={getattr(ev, 'source', 'unknown')}")

    # Event: Session end
    @session.on("session_end")
    def on_session_end(report):
        """
        Handle session end - trigger analysis and indexing.
        
        SessionReport contains:
        - session_id
        - duration
        - turns (list of conversation turns)
        """
        async def _run_post_session_tasks():
            logger.info(f"Session ended: {report.session_id if hasattr(report, 'session_id') else call_id}")
            logger.info(f"Total transcripts: {len(userdata.transcripts)}")

            # Run post-session tasks with timeout
            tasks = [
                trigger_call_end(call_id),
                trigger_rag_indexing(call_id, ward_id),
            ]

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=TIMEOUT_POST_SESSION,
                )
                logger.info("Post-session tasks completed")
            except asyncio.TimeoutError:
                logger.warning("Post-session tasks timed out")
            except Exception as e:
                logger.error(f"Post-session tasks failed: {e}")

        nonlocal post_session_task
        post_session_task = asyncio.create_task(_run_post_session_tasks())
        post_session_task.add_done_callback(lambda _t: session_end_event.set())

    # Create agent instance with call direction
    agent = ElderlyCompanionAgent(call_direction=call_direction)

    async def wait_for_bot_identity(timeout: float = 10.0) -> Optional[str]:
        """Wait for a bot participant to join, falling back to the first participant."""
        participants = list(ctx.room.remote_participants.values())
        for p in participants:
            if p.identity.startswith('bot-'):
                logger.info(f"Agent will listen to bot: {p.identity}")
                return p.identity
        if participants:
            logger.warning("Bot not found; using first participant")
            return participants[0].identity

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            participants = list(ctx.room.remote_participants.values())
            for p in participants:
                if p.identity.startswith('bot-'):
                    logger.info(f"Agent will listen to bot: {p.identity}")
                    return p.identity
    # Takeover state tracking - MUST be defined before event handlers
    takeover_active = False
    # session is already created above at line 171

    def check_admin_in_room():
        """Check if an admin is currently publishing audio in the room."""
        for p in ctx.room.remote_participants.values():
            if p.identity.startswith('admin_'):
                # Check if admin is publishing audio
                for pub in p.track_publications.values():
                    if pub.kind == 1 and pub.track:  # 1 = AUDIO
                        return True
        return False

    def update_takeover_state():
        """Update takeover state based on admin presence."""
        nonlocal takeover_active
        admin_present = check_admin_in_room()
        
        if admin_present and not takeover_active:
            takeover_active = True
            logger.info("🔴 ADMIN TAKEOVER DETECTED - Agent pausing")
            session.interrupt()
        elif not admin_present and takeover_active:
            takeover_active = False
            logger.info("🟢 ADMIN LEFT - Agent resuming")
            session.say("네, 다시 대화를 이어가겠습니다.")

    def log_all_participants():
        """Log all current participants for debugging."""
        logger.info(f"=== Current participants in room: {len(ctx.room.remote_participants)} ===")
        for p in ctx.room.remote_participants.values():
            tracks = []
            for pub in p.track_publications.values():
                tracks.append(f"{pub.kind}:{pub.sid}")
            logger.info(f"  - {p.identity} (sid={p.sid}, tracks={tracks})")

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant):
        """Handle new participant joining."""
        logger.info(f"👤 Participant connected: {participant.identity}")
        log_all_participants()
        if participant.identity.startswith('admin_'):
            logger.info("🔴 Admin joined room - preparing for takeover")
            update_takeover_state()

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        """Handle participant leaving."""
        logger.info(f"👤 Participant disconnected: {participant.identity}")
        if participant.identity.startswith('admin_'):
            logger.info("🟢 Admin left room - ending takeover")
            update_takeover_state()

    @ctx.room.on("track_published")
    def on_track_published(publication, participant):
        """Handle track publication - admin audio means takeover."""
        logger.info(f"🎙️ Track published: kind={publication.kind} from {participant.identity} (is_admin={participant.identity.startswith('admin_')})")
        if participant.identity.startswith('admin_') and publication.kind == 1:  # 1 = AUDIO
            logger.info(f"🔴 Admin started publishing audio: {participant.identity}")
            update_takeover_state()

    @ctx.room.on("track_unpublished")
    def on_track_unpublished(publication, participant):
        """Handle track unpublish - admin audio stop means takeover end."""
        logger.info(f"🎙️ Track unpublished: kind={publication.kind} from {participant.identity} (is_admin={participant.identity.startswith('admin_')})")
        if participant.identity.startswith('admin_') and publication.kind == 1:  # 1 = AUDIO
            logger.info(f"🟢 Admin stopped publishing audio: {participant.identity}")
            update_takeover_state()

    @ctx.room.on("room_metadata_changed")
    def on_room_metadata_changed(old_metadata, new_metadata):
        """Handle room metadata changes - PRIMARY takeover detection mechanism."""
        nonlocal takeover_active
        try:
            logger.info(f"📋 Room metadata changed: {new_metadata}")
            import json
            data = json.loads(new_metadata) if new_metadata else {}
            
            if data.get("takeover") and not takeover_active:
                takeover_active = True
                logger.info("🔴 METADATA: Takeover started - Agent pausing")
                # Stop all agent processing
                session.interrupt()
                session.clear_user_turn()  # Clear any pending user input
                logger.info("🔴 Agent fully paused - speech and input cleared")
            elif not data.get("takeover") and takeover_active:
                takeover_active = False
                logger.info("🟢 METADATA: Takeover ended - Agent resuming")
                session.say("네, 다시 대화를 이어가겠습니다.")
        except Exception as e:
            logger.error(f"Error processing room metadata: {e}")

    logger.info("Room event handlers registered for takeover detection")

    # Wait for participant to join
    await asyncio.sleep(3)

    bot_identity = await wait_for_bot_identity()
    # Find target participant to listen to
    # Priority: 1) bot-* participant, 2) first non-admin participant, 3) None (listen to all)
    target_identity = None
    logger.info(f"Participants in room: {len(ctx.room.remote_participants)}")
    for p in ctx.room.remote_participants.values():
        logger.info(f"  - {p.identity}")
        if p.identity.startswith('bot-'):
            target_identity = p.identity
            logger.info(f"Agent will listen to bot: {target_identity}")
            break
        elif not p.identity.startswith('admin_') and target_identity is None:
            # First non-admin participant as fallback
            target_identity = p.identity
            logger.info(f"Agent will listen to user: {target_identity}")

    if not target_identity:
        logger.warning("No suitable participant found - agent will listen to all participants")

    # Start session with target participant
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            close_on_disconnect=False,
            participant_identity=target_identity,  # Listen to target (or all if None)
        ),
    )

    # User input handler that respects takeover state
    @session.on("user_input_transcribed")
    def on_user_transcript_with_takeover(ev):
        """Capture user transcripts but ignore during takeover."""
        if takeover_active:
            logger.info(f"⏸️ Ignoring user input during takeover: {ev.transcript}")
            return
        if ev.is_final:
            userdata.add_transcript("user", ev.transcript)
            logger.info(f"👤 User transcript: {ev.transcript}")
            # Broadcast to frontend
            asyncio.create_task(broadcast_transcript("user", ev.transcript))
        # Note: AgentSession automatically handles LLM response generation

    # Greeting
    session.say("안녕하세요, 어르신. 오늘 어떻게 지내셨어요?")

    logger.info(f"Agent ready and listening in room: {ctx.room.name}")

    # Start async polling task to detect admin presence (fallback for missed events)
    async def poll_for_admin():
        """Periodically check for admin presence since events may not fire."""
        nonlocal takeover_active
        last_state = False
        while True:
            try:
                await asyncio.sleep(2)  # Check every 2 seconds
                
                # Check for admin with audio
                admin_present = False
                for p in ctx.room.remote_participants.values():
                    if p.identity.startswith('admin_'):
                        logger.info(f"🔍 Polling: Found admin {p.identity} with {len(p.track_publications)} tracks")
                        for pub in p.track_publications.values():
                            if pub.kind == 1:  # AUDIO
                                admin_present = True
                                break
                
                # Detect state change
                if admin_present and not last_state:
                    logger.info("🔴 POLLING: Admin audio detected - pausing agent")
                    takeover_active = True
                    session.interrupt()
                elif not admin_present and last_state:
                    logger.info("🟢 POLLING: Admin audio gone - resuming agent")
                    takeover_active = False
                    session.say("네, 다시 대화를 이어가겠습니다.")
                
                last_state = admin_present
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")

    # Start the polling task
    poll_task = asyncio.create_task(poll_for_admin())
    logger.info("Admin detection polling started")

    # Keep the job alive until post-session tasks finish.
    await session_end_event.wait()


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
