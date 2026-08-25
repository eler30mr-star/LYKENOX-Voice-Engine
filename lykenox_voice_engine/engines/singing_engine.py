"""Singing synthesis backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lykenox_voice_engine.models.notes import NoteEvent


class SingingVoiceEngine(ABC):
    """Abstract interface for trainable score-to-singing backends."""

    @abstractmethod
    def check_available(self) -> dict[str, Any]:
        """Return backend availability, versions, and blocking errors."""

    @abstractmethod
    def prepare_dataset(self, profile: str) -> dict[str, Any]:
        """Prepare a microtest dataset without modifying raw originals."""

    @abstractmethod
    def train(self, profile: str) -> dict[str, Any]:
        """Run the smallest valid backend training job."""

    @abstractmethod
    def resume_training(self, profile: str, checkpoint: str) -> dict[str, Any]:
        """Resume training from a valid checkpoint."""

    @abstractmethod
    def synthesize(self, profile: str, lyrics: str, notes: list[NoteEvent], tempo: int) -> Path:
        """Generate a sung waveform from lyrics and score data."""

    @abstractmethod
    def cancel(self, job_id: str | None = None) -> None:
        """Cancel active backend processes."""

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return selected model/checkpoint metadata."""


SingingEngine = SingingVoiceEngine
