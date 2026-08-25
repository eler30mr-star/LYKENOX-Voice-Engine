"""Synthesis job state models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class JobStatus(StrEnum):
    """Allowed asynchronous job states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VoiceJob:
    """Track one local synthesis job."""

    id: str
    status: JobStatus
    output_path: str | None = None
    error: str | None = None
