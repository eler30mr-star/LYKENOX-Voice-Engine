"""Singing synthesis backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from lykenox_voice_engine.models.note import NoteSequence


@dataclass(frozen=True)
class SynthesisRequest:
    """Backend-neutral singing synthesis request."""

    profile_id: str
    lyrics: str
    notes: NoteSequence
    output_format: str
    pitch_transpose: int = 0


class SingingEngine(ABC):
    """Abstract score-to-singing backend."""

    @abstractmethod
    def synthesize(self, request: SynthesisRequest, output_dir: Path) -> Path:
        """Synthesize a vocal waveform and return the output path."""
