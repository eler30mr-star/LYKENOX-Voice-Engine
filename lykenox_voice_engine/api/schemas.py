"""FastAPI request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NoteItem(BaseModel):
    """One note event in API JSON."""

    lyric: str
    midi: int = Field(ge=0, le=127)
    start: float = Field(ge=0)
    duration: float = Field(gt=0)


class SynthesizeRequest(BaseModel):
    """Legacy WORLDLINE/voicebank singing request."""

    profile: str
    lyrics: str
    tempo: int = Field(gt=0)
    notes: list[NoteItem]
    output_format: str = "wav"
    f0: str = "score"
    version: str = "worldline-legacy"


class SpeakRequest(BaseModel):
    """Direct text-to-speech request for the personal LYKENOX identity model."""

    profile: str = "lykenox"
    text: str = Field(min_length=1)
    language: str = "es"
    speaking_rate: float = Field(default=1.0, gt=0.25, lt=4.0)
    output_format: str = "wav"


class SingRequest(BaseModel):
    """Direct text-to-singing request for the personal LYKENOX identity model."""

    profile: str = "lykenox"
    lyrics: str = Field(min_length=1)
    tempo: int = Field(gt=0)
    notes: list[NoteItem]
    language: str = "es"
    output_format: str = "wav"


class SynthesizeMidiRequest(BaseModel):
    """Request for MIDI-based singing synthesis."""

    profile: str
    lyrics: str
    midi_path: str
    tempo: int = Field(gt=0)
    output_format: str = "wav"


class JobResponse(BaseModel):
    """Standard job response body."""

    job_id: str
    status: str
    output_path: str | None = None
    error: str | None = None
