"""In-memory local job registry for Phase 1 API calls."""

from __future__ import annotations

import uuid
from threading import Lock

from lykenox_voice_engine.models.job import JobStatus, VoiceJob


class JobRegistry:
    """Store synthesis job state for the local FastAPI service."""

    def __init__(self) -> None:
        self._jobs: dict[str, VoiceJob] = {}
        self._lock = Lock()

    def create(self) -> VoiceJob:
        """Create a queued job and return it."""

        job = VoiceJob(id=str(uuid.uuid4()), status=JobStatus.QUEUED)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> VoiceJob | None:
        """Return a job by id, if known."""

        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> VoiceJob | None:
        """Mark a queued or running job as cancelled."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job.status = JobStatus.CANCELLED
            return job
