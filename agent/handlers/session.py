"""Session lifecycle handling."""
import asyncio
import json
import logging
from typing import Optional

from livekit.agents import JobContext

from services.redis_pubsub import publish_call_end
from services.api_client import notify_call_end

logger = logging.getLogger(__name__)


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

    logger.info(f"Using room name as ward_id: {room_name}")
    return room_name


def parse_metadata(raw) -> dict:
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
    return parse_metadata(raw)


async def wait_for_participant(ctx: JobContext, timeout: float = 10.0) -> Optional[str]:
    """Wait for a non-admin participant (iOS user) to join."""

    def find_target(participants) -> Optional[str]:
        """Find first non-admin participant."""
        for p in participants:
            if not p.identity.startswith("admin"):
                return p.identity
        return None

    # Check existing participants
    participants = list(ctx.room.remote_participants.values())
    target = find_target(participants)
    if target:
        logger.info(f"Found target participant: {target}")
        return target

    # Wait for participant
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.5)
        participants = list(ctx.room.remote_participants.values())
        target = find_target(participants)
        if target:
            logger.info(f"Found target participant: {target}")
            return target

    logger.warning("No non-admin participant found within timeout")
    return None


def get_target_identity(ctx: JobContext) -> Optional[str]:
    """Get the target participant identity to listen to."""
    target_identity = None
    for p in ctx.room.remote_participants.values():
        if p.identity.startswith('bot-'):
            return p.identity
        elif not p.identity.startswith('admin_') and target_identity is None:
            target_identity = p.identity
    return target_identity


class SessionEndHandler:
    """Handles session end events."""

    def __init__(self, call_id: str, ward_id: str):
        self.call_id = call_id
        self.ward_id = ward_id
        self.session_end_event = asyncio.Event()
        self._post_session_task = None

    async def _run_post_session_tasks(self, report):
        """Run cleanup tasks after session ends."""
        session_id = report.session_id if hasattr(report, 'session_id') else self.call_id
        logger.info(f"Session ended: {session_id}")

        # Publish call_end event
        try:
            success = await publish_call_end(self.call_id, self.ward_id)
            if not success:
                await notify_call_end(self.call_id, self.ward_id)
            logger.info("Call end event published")
        except Exception as e:
            logger.error(f"Failed to publish call_end event: {e}")

    def handle_session_end(self, report):
        """Handle session end event."""
        self._post_session_task = asyncio.create_task(self._run_post_session_tasks(report))
        self._post_session_task.add_done_callback(lambda _: self.session_end_event.set())

    async def wait_for_completion(self):
        """Wait for post-session tasks to complete."""
        await self.session_end_event.wait()
