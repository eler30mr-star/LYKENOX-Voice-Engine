"""Synthesis service and internal melody format."""

from dataclasses import dataclass
from pathlib import Path

from lykenox_voice_engine.engines.audit_only import AuditOnlyEngine
from lykenox_voice_engine.engines.singing_engine import SingingVoiceEngine
from lykenox_voice_engine.models.notes import NoteEvent


@dataclass(frozen=True)
class SynthesisRequest:
    """Local score-to-singing request."""

    profile: str
    lyrics: str
    notes: list[NoteEvent]
    tempo: int
    output_format: str = "wav"


class SynthesisService:
    """Coordinate singing synthesis with the selected backend."""

    def __init__(self, engine: SingingVoiceEngine | None = None) -> None:
        """Create service with a safe audit-only default."""
        self.engine = engine or AuditOnlyEngine()

    def synthesize(self, request: SynthesisRequest) -> Path:
        """Generate singing, refusing to fake output when backend is missing."""
        return self.engine.synthesize(request.profile, request.lyrics, request.notes, request.tempo)


def simple_test_melody() -> list[NoteEvent]:
    """Return a simple built-in Spanish melody for validation."""
    syllables = ["Bai", "lan", "do", "te", "co", "no", "ci", "ven"]
    notes = [64, 64, 67, 67, 69, 69, 67, 65]
    return [NoteEvent(s, m, i * 0.5, 0.45) for i, (s, m) in enumerate(zip(syllables, notes))]
