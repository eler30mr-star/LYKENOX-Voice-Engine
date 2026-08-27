"""Backend-independent contracts for LYKENOX-owned voice models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from lykenox_voice_engine.models.notes import NoteEvent


class VoiceMode(StrEnum):
    """Direct generation modes owned by the LYKENOX product contract."""

    SPEECH = "speech"
    SINGING = "singing"


class ModelStatus(StrEnum):
    """Lifecycle state for a persistent LYKENOX identity model."""

    NOT_TRAINED = "not_trained"
    DATASET_READY = "dataset_ready"
    TRAINING = "training"
    READY = "ready"
    INCOMPATIBLE = "incompatible"


class ArtifactFormat(StrEnum):
    """Runtime artifact formats supported by the product layer.

    These are execution formats, not third-party TTS product identities.
    """

    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    LYKENOX_NATIVE = "lykenox_native"


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    """Direct text-to-speech request for a persistent LYKENOX voice."""

    profile: str
    text: str
    language: str = "es"
    speaking_rate: float = 1.0


@dataclass(frozen=True)
class SingingSynthesisRequest:
    """Direct score-conditioned singing request for a persistent LYKENOX voice."""

    profile: str
    lyrics: str
    notes: list[NoteEvent]
    tempo: int
    language: str = "es"


@dataclass(frozen=True)
class IdentityModelManifest:
    """Persistent metadata for one LYKENOX-owned model artifact.

    ``architecture_family`` records the neural design lineage for reproducibility.
    It must never be used as a runtime dependency on another TTS product.
    ``trainer_provenance`` records how an artifact was produced; inference must be
    self-contained and must not require that training project to be installed.
    """

    profile: str
    owner_identity: str
    language: str
    status: ModelStatus
    supports_speech: bool
    supports_singing: bool
    uses_source_singer: bool
    uses_voice_conversion: bool
    requires_reference_audio: bool
    training_dataset: str
    artifact_format: ArtifactFormat = ArtifactFormat.ONNX
    architecture_family: str = "unselected"
    trainer_provenance: str = "lykenox"
    runtime_entrypoint: str = ""
    model_files: tuple[str, ...] = field(default_factory=tuple)
    product_contract_version: int = 1


def default_manifest(profile: str = "lykenox") -> IdentityModelManifest:
    """Return the backend-independent manifest before a neural model exists."""

    return IdentityModelManifest(
        profile=profile,
        owner_identity="LYKENOX original voice",
        language="es",
        status=ModelStatus.NOT_TRAINED,
        supports_speech=False,
        supports_singing=False,
        uses_source_singer=False,
        uses_voice_conversion=False,
        requires_reference_audio=False,
        training_dataset=f"datasets/{profile}/identity_voice",
    )
