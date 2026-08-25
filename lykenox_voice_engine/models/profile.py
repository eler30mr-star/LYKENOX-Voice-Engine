"""Profile metadata models."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class VoiceProfile:
    """Persisted singing voice identity profile."""

    id: str = "lykenox"
    name: str = "LYKENOX Voice"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    dataset_duration: float = 0.0
    clips: int = 0
    model_type: str = "unselected"
    sample_rate: int = 0
    training_steps: int = 0
    training_epochs: int = 0
    checkpoint: str = ""
    speaker_embedding: str = ""
    status: str = "auditing"

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "VoiceProfile":
        """Build a profile from persisted JSON data."""
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})
