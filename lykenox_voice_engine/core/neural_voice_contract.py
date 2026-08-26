"""Contracts for the LYKENOX identity voice model target."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lykenox_voice_engine.models.notes import NoteEvent


class VoiceMode(StrEnum):
    """Supported direct generation modes for the identity model."""

    SPEECH = "speech"
    SINGING = "singing"


class ModelStatus(StrEnum):
    """Lifecycle state for the personal LYKENOX model."""

    NOT_TRAINED = "not_trained"
    DATASET_READY = "dataset_ready"
    TRAINING = "training"
    READY = "ready"


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    """Direct text-to-speech request for the personal LYKENOX voice."""

    profile: str
    text: str
    language: str = "es"
    speaking_rate: float = 1.0


@dataclass(frozen=True)
class SingingSynthesisRequest:
    """Direct text-to-singing request for the personal LYKENOX voice."""

    profile: str
    lyrics: str
    notes: list[NoteEvent]
    tempo: int
    language: str = "es"


@dataclass(frozen=True)
class IdentityModelManifest:
    """Persistent intent and readiness metadata for one personal voice model."""

    profile: str
    owner_identity: str
    language: str
    status: ModelStatus
    supports_speech: bool
    supports_singing: bool
    uses_source_singer: bool
    uses_voice_conversion: bool
    training_dataset: str


def default_manifest(profile: str = "lykenox") -> IdentityModelManifest:
    """Return the expected manifest before the neural model is trained."""

    return IdentityModelManifest(
        profile=profile,
        owner_identity="LYKENOX original voice",
        language="es",
        status=ModelStatus.NOT_TRAINED,
        supports_speech=False,
        supports_singing=False,
        uses_source_singer=False,
        uses_voice_conversion=False,
        training_dataset=f"datasets/{profile}/identity_voice",
    )
