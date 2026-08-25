"""Internal note representation for score-to-singing synthesis."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NoteEvent:
    """One lyric syllable/note event for singing synthesis."""

    lyric: str
    midi: int
    start: float
    duration: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the note event."""
        return asdict(self)
