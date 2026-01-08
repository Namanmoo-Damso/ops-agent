"""Session userdata for voice agent."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.elderly_companion import CallDirection

logger = logging.getLogger(__name__)

# Maximum number of transcripts to keep in memory
MAX_TRANSCRIPTS = 500


@dataclass
class TranscriptEntry:
    """Single transcript entry."""
    speaker_type: str  # 'user' or 'agent'
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    is_final: bool = True


@dataclass
class SessionUserdata:
    """
    Session-level userdata for the voice agent.

    Stored in AgentSession and accessible via RunContext.
    """
    ward_id: str
    call_id: str
    call_direction: str = "inbound"  # "inbound" or "outbound"
    transcripts: list[TranscriptEntry] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)

    def add_transcript(self, speaker_type: str, text: str, is_final: bool = True):
        """Add a transcript entry with memory limit."""
        self.transcripts.append(TranscriptEntry(
            speaker_type=speaker_type,
            text=text,
            is_final=is_final,
        ))

        # Prevent memory leak by limiting transcript count
        if len(self.transcripts) > MAX_TRANSCRIPTS:
            self.transcripts = self.transcripts[-MAX_TRANSCRIPTS:]
            logger.warning(f"Transcript limit reached, keeping last {MAX_TRANSCRIPTS}")

    def get_user_transcripts(self) -> list[str]:
        """Get all user transcripts as text."""
        return [t.text for t in self.transcripts if t.speaker_type == 'user' and t.is_final]

    def get_full_transcript(self) -> str:
        """Get full conversation as formatted text."""
        lines = []
        for t in self.transcripts:
            if t.is_final:
                speaker = "어르신" if t.speaker_type == 'user' else "AI"
                lines.append(f"{speaker}: {t.text}")
        return "\n".join(lines)
