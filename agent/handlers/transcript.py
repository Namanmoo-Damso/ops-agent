"""Transcript handling - publishing and broadcasting."""
import asyncio
import json
import logging
from typing import Set, Callable

from services.redis_pubsub import publish_transcript, store_transcript_direct

logger = logging.getLogger(__name__)


class TranscriptHandler:
    """Handles transcript publishing and broadcasting."""

    def __init__(self, call_id: str, room, userdata):
        self.call_id = call_id
        self.room = room
        self.userdata = userdata
        self._background_tasks: Set[asyncio.Task] = set()

    def _create_tracked_task(self, coro) -> asyncio.Task:
        """Create a task and track it to prevent memory leaks."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def publish_transcript_event(self, speaker: str, text: str):
        """Publish transcript event to Redis with fallback."""
        success = await publish_transcript(self.call_id, speaker, text)
        if not success:
            await store_transcript_direct(self.call_id, speaker, text)

    async def broadcast_to_room(self, role: str, text: str):
        """Broadcast transcript to room via data packet."""
        try:
            payload = json.dumps({
                "type": "transcript",
                "role": role,
                "text": text,
                "timestamp": int(asyncio.get_event_loop().time() * 1000)
            })
            await self.room.local_participant.publish_data(
                payload,
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to broadcast transcript: {e}")

    def handle_user_transcript(self, ev, takeover_active: bool = False):
        """Handle user transcript event."""
        if takeover_active:
            return
        if ev.is_final:
            self.userdata.add_transcript("user", ev.transcript)
            logger.debug(f"User transcript: {ev.transcript}")
            self._create_tracked_task(self.publish_transcript_event("user", ev.transcript))
            self._create_tracked_task(self.broadcast_to_room("user", ev.transcript))

    def handle_agent_speech(self, ev):
        """Handle agent speech committed event."""
        if hasattr(ev, 'content') and ev.content:
            self.userdata.add_transcript("agent", ev.content)
            logger.debug(f"Agent response: {ev.content}")
            self._create_tracked_task(self.publish_transcript_event("agent", ev.content))

    def handle_conversation_item(self, ev):
        """Handle conversation item added event."""
        try:
            item = ev.item if hasattr(ev, 'item') else ev
            if hasattr(item, 'role') and item.role == 'assistant':
                content = self._extract_content(item)
                if content and content.strip():
                    self.userdata.add_transcript("agent", content)
                    self._create_tracked_task(self.publish_transcript_event("agent", content))
                    self._create_tracked_task(self.broadcast_to_room("agent", content))
        except Exception as e:
            logger.error(f"Error in conversation_item_added handler: {e}")

    def _extract_content(self, item) -> str:
        """Extract text content from conversation item."""
        if hasattr(item, 'text') and item.text:
            return item.text
        elif hasattr(item, 'content'):
            raw_content = item.content
            if isinstance(raw_content, str):
                return raw_content
            elif isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif hasattr(part, 'text'):
                        text_parts.append(part.text)
                    elif hasattr(part, '__str__'):
                        text_parts.append(str(part))
                return ' '.join(text_parts)
            else:
                return str(raw_content)
        return ""
