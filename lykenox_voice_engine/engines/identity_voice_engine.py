"""Identity voice engine facade for future neural speech and singing models."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lykenox_voice_engine.core.neural_voice_contract import (
    IdentityModelManifest,
    SingingSynthesisRequest,
    SpeechSynthesisRequest,
    default_manifest,
)


class IdentityModelUnavailableError(RuntimeError):
    """Raised when the personal neural voice model is not ready."""


class IdentityVoiceEngine:
    """Facade for direct generation with the user's own trained voice identity."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "models" / "lykenox_identity" / "manifest.json"

    def status(self) -> dict[str, Any]:
        """Return current neural identity model readiness."""

        manifest = self.load_or_create_manifest()
        return {
            **asdict(manifest),
            "manifest_path": str(self.manifest_path),
            "objective": "direct speech and singing synthesis with LYKENOX identity",
            "not_allowed": ["source singer", "RVC", "SVC primary architecture", "voice conversion"],
        }

    def load_or_create_manifest(self) -> IdentityModelManifest:
        """Load the model manifest, creating a not-trained manifest when absent."""

        if not self.manifest_path.exists():
            manifest = default_manifest()
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
            return manifest
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return IdentityModelManifest(**data)

    def synthesize_speech(self, request: SpeechSynthesisRequest, output_path: Path) -> Path:
        """Generate speech with the trained identity model, or fail honestly."""

        manifest = self.load_or_create_manifest()
        if not manifest.supports_speech:
            raise IdentityModelUnavailableError(
                "Modelo neural LYKENOX para lectura de texto no entrenado todavía."
            )
        raise IdentityModelUnavailableError("Runtime neural de speech aun no integrado.")

    def synthesize_singing(self, request: SingingSynthesisRequest, output_path: Path) -> Path:
        """Generate singing with the trained identity model, or fail honestly."""

        manifest = self.load_or_create_manifest()
        if not manifest.supports_singing:
            raise IdentityModelUnavailableError(
                "Modelo neural LYKENOX para canto directo no entrenado todavía."
            )
        raise IdentityModelUnavailableError("Runtime neural de singing aun no integrado.")
