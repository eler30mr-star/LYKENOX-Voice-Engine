"""Internal note format for score-to-singing requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoteEvent:
    """A lyric token aligned to a MIDI pitch and duration."""

    lyric: str
    midi: int
    start: float
    duration: float

    def validate(self) -> None:
        """Validate pitch and timing values for one note event."""

        if not self.lyric:
            raise ValueError("note lyric is required")
        if self.midi < 0 or self.midi > 127:
            raise ValueError("midi must be between 0 and 127")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")


@dataclass(frozen=True)
class NoteSequence:
    """Tempo plus ordered note events for singing synthesis."""

    tempo: int
    notes: tuple[NoteEvent, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteSequence":
        """Create a note sequence from API JSON payload data."""

        sequence = cls(
            tempo=int(data["tempo"]),
            notes=tuple(NoteEvent(**item) for item in data["notes"]),
        )
        sequence.validate()
        return sequence

    def validate(self) -> None:
        """Validate tempo and contained notes."""

        if self.tempo <= 0:
            raise ValueError("tempo must be positive")
        if not self.notes:
            raise ValueError("at least one note is required")
        for note in self.notes:
            note.validate()
