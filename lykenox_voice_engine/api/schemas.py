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
    """Request for direct singing synthesis from lyrics and notes."""

    profile: str
    lyrics: str
    tempo: int = Field(gt=0)
    notes: list[NoteItem]
    output_format: str = "wav"
    pitch: int = 0
    index_rate: float = 0.0
    protect: float = 0.33
    f0: str = "score"
    version: str = "svs-v1"


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
