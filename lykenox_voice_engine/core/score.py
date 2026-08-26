"""Score file loading for local LYKENOX singing synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lykenox_voice_engine.models.notes import NoteEvent


@dataclass(frozen=True)
class LykenoxScore:
    """A saved lyrics, tempo, and note score for direct local singing."""

    title: str
    profile: str
    lyrics: str
    tempo: int
    notes: tuple[NoteEvent, ...]

    def note_rows(self) -> list[dict[str, Any]]:
        """Return note rows suitable for JSON and UI display."""

        return [
            {
                "lyric": note.lyric,
                "midi": note.midi,
                "start": note.start,
                "duration": note.duration,
            }
            for note in self.notes
        ]


def load_score(path: Path) -> LykenoxScore:
    """Load one LYKENOX score JSON file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    notes = tuple(NoteEvent(**item) for item in data.get("notes", []))
    return LykenoxScore(
        title=str(data.get("title", path.stem)),
        profile=str(data.get("profile", "lykenox")),
        lyrics=str(data["lyrics"]),
        tempo=int(data["tempo"]),
        notes=notes,
    )


def score_to_json(score: LykenoxScore) -> str:
    """Serialize a score payload for the existing editor."""

    return json.dumps(
        {
            "tempo": score.tempo,
            "notes": score.note_rows(),
        },
        indent=2,
        ensure_ascii=False,
    )
