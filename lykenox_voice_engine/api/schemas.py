"""FastAPI request and response schemas."""

from pydantic import BaseModel, Field


class NoteSchema(BaseModel):
    """One note event in the public API."""

    lyric: str
    midi: int = Field(ge=0, le=127)
    start: float = Field(ge=0)
    duration: float = Field(gt=0)


class SynthesizeRequest(BaseModel):
    """Score-to-singing synthesis request."""

    profile: str = "lykenox"
    lyrics: str
    tempo: int = Field(gt=0, le=300)
    notes: list[NoteSchema]
    output_format: str = "wav"


class JobResponse(BaseModel):
    """API job response."""

    job_id: str
    status: str
    output_path: str = ""
