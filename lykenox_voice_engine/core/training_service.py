"""Local identity readiness controls.

This project runs on the user's CPU-only Windows machine. The active local route
is a personal voicebank engine; discarded neural/SVS runtimes must not be exposed
as trainable backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lykenox_voice_engine.core.identity_dataset import IdentityDatasetService
from lykenox_voice_engine.core.voicebank import VoicebankManager


class TrainingService:
    """Report local readiness without starting unsupported neural training."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.identity = IdentityDatasetService(self.root)
        self.voicebank = VoicebankManager(self.root)

    def check(self) -> dict[str, Any]:
        """Return readiness of the local voicebank and identity dataset."""

        return {
            "ok": True,
            "backend": "lykenox_local_voicebank",
            "device": "cpu",
            "neural_training": "disabled",
            "reason": "Ruta activa: grabaciones propias + voicebank local + WORLDLINE-R.",
            "identity_dataset": self.identity.summary(),
            "voicebank": self.voicebank.validate_voicebank(),
        }

    def prepare_microtest(self) -> dict[str, Any]:
        """Prepare local recording folders only."""

        self.identity.ensure()
        self.voicebank.raw_dir.mkdir(parents=True, exist_ok=True)
        self.voicebank.wav_dir.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "status": "ready_for_local_recording",
            "identity_dataset": str(self.identity.base_dir),
            "voicebank_raw": str(self.voicebank.raw_dir),
            "voicebank": str(self.voicebank.voicebank_dir),
        }

    def microtest(self) -> dict[str, Any]:
        """Refuse unsupported neural microtraining."""

        return {
            "ok": False,
            "backend": "lykenox_local_voicebank",
            "reason": "Entrenamiento neural descartado en esta maquina CPU; usar microtest multipitch.",
        }

    def stop(self) -> dict[str, str]:
        """Return a no-op stop response for the local route."""

        return {"status": "idle"}
