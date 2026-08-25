"""Voice profile metadata models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceProfile:
    """Persistent vocal identity profile metadata."""

    id: str
    name: str
    dataset_duration: float
    clips: int
    model_type: str
    sample_rate: int
    training_steps: int
    training_epochs: int
    checkpoint: str | None
    speaker_embedding: str | None
    status: str

    @classmethod
    def load(cls, path: Path) -> "VoiceProfile":
        """Load a profile from profile.json."""

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__.keys()})
