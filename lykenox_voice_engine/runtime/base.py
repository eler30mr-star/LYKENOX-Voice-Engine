"""Runtime interfaces owned by LYKENOX Voice Engine.

A packaged product runtime must load LYKENOX model artifacts directly. It must
not shell out to Piper, Coqui, OpenUtau, or any other external TTS/SVS product.
Infrastructure libraries such as ONNX Runtime may be embedded as redistributable
implementation dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lykenox_voice_engine.core.neural_voice_contract import (
    IdentityModelManifest,
    SingingSynthesisRequest,
    SpeechSynthesisRequest,
)


class LykenoxRuntime(ABC):
    """Stable product-facing interface for persistent identity inference."""

    @abstractmethod
    def load(self, model_dir: Path, manifest: IdentityModelManifest) -> None:
        """Load a self-contained LYKENOX artifact from disk."""

    @abstractmethod
    def health(self) -> dict[str, object]:
        """Return runtime and artifact readiness without external network access."""


class LykenoxSpeechRuntime(LykenoxRuntime):
    """Direct text -> LYKENOX speech runtime."""

    @abstractmethod
    def synthesize(self, request: SpeechSynthesisRequest, output_path: Path) -> Path:
        """Generate speech without reference audio or a third-party TTS executable."""


class LykenoxSingingRuntime(LykenoxRuntime):
    """Direct lyrics + score -> LYKENOX singing runtime."""

    @abstractmethod
    def synthesize(self, request: SingingSynthesisRequest, output_path: Path) -> Path:
        """Generate singing without source-singer conversion."""
