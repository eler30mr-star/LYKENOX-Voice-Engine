"""Singing voice engine abstraction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lykenox_voice_engine.models.notes import NoteEvent


class SingingVoiceEngine(ABC):
    """Backend interface for score-to-singing synthesis engines."""

    @abstractmethod
    def check_available(self) -> dict[str, Any]:
        """Return runtime availability and limitations."""

    @abstractmethod
    def prepare_dataset(self, profile: str) -> dict[str, Any]:
        """Prepare a dataset for this backend."""

    @abstractmethod
    def train(self, profile: str) -> dict[str, Any]:
        """Start or run training for the selected profile."""

    @abstractmethod
    def resume_training(self, profile: str, checkpoint: str) -> dict[str, Any]:
        """Resume training from a backend checkpoint."""

    @abstractmethod
    def synthesize(self, profile: str, lyrics: str, notes: list[NoteEvent], tempo: int) -> Path:
        """Synthesize singing from lyrics, notes, and tempo."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a running job if supported."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return backend model metadata."""
