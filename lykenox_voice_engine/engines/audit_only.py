"""Placeholder engine used until a real SVS backend is selected."""

from pathlib import Path
from typing import Any

from lykenox_voice_engine.engines.singing_engine import SingingVoiceEngine
from lykenox_voice_engine.models.notes import NoteEvent


class AuditOnlyEngine(SingingVoiceEngine):
    """Non-synthesizing backend that prevents fake voice generation."""

    def check_available(self) -> dict[str, Any]:
        """Report that backend selection is still pending."""
        return {"available": False, "reason": "Motor SVS no seleccionado todavía."}

    def prepare_dataset(self, profile: str) -> dict[str, Any]:
        """Refuse backend preparation until a real engine is selected."""
        return {"ok": False, "profile": profile, "reason": "Auditoría técnica pendiente."}

    def train(self, profile: str) -> dict[str, Any]:
        """Refuse training in the scaffold phase."""
        return {"ok": False, "profile": profile, "reason": "No hay backend entrenable instalado."}

    def resume_training(self, profile: str, checkpoint: str) -> dict[str, Any]:
        """Refuse resume in the scaffold phase."""
        return {"ok": False, "profile": profile, "checkpoint": checkpoint}

    def synthesize(self, profile: str, lyrics: str, notes: list[NoteEvent], tempo: int) -> Path:
        """Raise instead of producing fake singing."""
        raise RuntimeError("No hay motor SVS real seleccionado; no se genera audio falso.")

    def cancel(self, job_id: str) -> None:
        """No-op because no backend job is running."""

    def get_model_info(self) -> dict[str, Any]:
        """Return pending backend information."""
        return {"backend": "audit-only", "selected": False}
