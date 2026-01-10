"""Admin takeover handling."""
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class TakeoverHandler:
    """Handles admin takeover detection and state management."""

    def __init__(self, room, session):
        self.room = room
        self.session = session
        self.takeover_active = False
        self._poll_task = None

    def check_admin_in_room(self) -> bool:
        """Check if an admin is currently publishing audio in the room."""
        for p in self.room.remote_participants.values():
            if p.identity.startswith('admin_'):
                for pub in p.track_publications.values():
                    if pub.kind == 1 and pub.track:  # 1 = AUDIO
                        return True
        return False

    def update_takeover_state(self):
        """Update takeover state based on admin presence."""
        admin_present = self.check_admin_in_room()

        if admin_present and not self.takeover_active:
            self.takeover_active = True
            logger.info("ADMIN TAKEOVER DETECTED - Agent pausing")
            self.session.interrupt()
        elif not admin_present and self.takeover_active:
            self.takeover_active = False
            self.session.say("네, 다시 대화를 이어가겠습니다.")

    def handle_participant_connected(self, participant):
        """Handle new participant joining."""
        if participant.identity.startswith('admin_'):
            self.update_takeover_state()

    def handle_participant_disconnected(self, participant):
        """Handle participant leaving."""
        if participant.identity.startswith('admin_'):
            self.update_takeover_state()

    def handle_track_published(self, publication, participant):
        """Handle track publication - admin audio means takeover."""
        if participant.identity.startswith('admin_') and publication.kind == 1:
            self.update_takeover_state()

    def handle_track_unpublished(self, publication, participant):
        """Handle track unpublish - admin audio stop means takeover end."""
        if participant.identity.startswith('admin_') and publication.kind == 1:
            self.update_takeover_state()

    def handle_room_metadata_changed(self, old_metadata, new_metadata):
        """Handle room metadata changes - PRIMARY takeover detection mechanism."""
        try:
            data = json.loads(new_metadata) if new_metadata else {}

            if data.get("takeover") and not self.takeover_active:
                self.takeover_active = True
                self.session.interrupt()
                self.session.clear_user_turn()
            elif not data.get("takeover") and self.takeover_active:
                self.takeover_active = False
                self.session.say("네, 다시 대화를 이어가겠습니다.")
        except Exception as e:
            logger.error(f"Error processing room metadata: {e}")

    def register_event_handlers(self):
        """Register all room event handlers for takeover detection."""
        self.room.on("participant_connected")(self.handle_participant_connected)
        self.room.on("participant_disconnected")(self.handle_participant_disconnected)
        self.room.on("track_published")(self.handle_track_published)
        self.room.on("track_unpublished")(self.handle_track_unpublished)
        self.room.on("room_metadata_changed")(self.handle_room_metadata_changed)

    async def start_polling(self):
        """Start polling for admin presence (backup mechanism)."""
        self._poll_task = asyncio.create_task(self._poll_for_admin())

    async def _poll_for_admin(self):
        """Periodically check for admin presence since events may not fire."""
        last_state = False
        while True:
            try:
                await asyncio.sleep(2)
                admin_present = self.check_admin_in_room()

                if admin_present and not last_state:
                    self.takeover_active = True
                    self.session.interrupt()
                elif not admin_present and last_state:
                    self.takeover_active = False
                    self.session.say("네, 다시 대화를 이어가겠습니다.")

                last_state = admin_present
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")

    def stop_polling(self):
        """Stop the polling task."""
        if self._poll_task:
            self._poll_task.cancel()

    def log_all_participants(self):
        """Log all current participants for debugging."""
        logger.info(f"=== Current participants in room: {len(self.room.remote_participants)} ===")
        for p in self.room.remote_participants.values():
            tracks = [f"{pub.kind}:{pub.sid}" for pub in p.track_publications.values()]
            logger.info(f"  - {p.identity} (sid={p.sid}, tracks={tracks})")
