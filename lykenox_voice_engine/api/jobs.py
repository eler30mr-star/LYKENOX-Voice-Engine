"""In-process asynchronous job registry for local API calls."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class Job:
    """Represent one local synthesis job."""

    id: str
    status: str = "queued"
    progress: float = 0.0
    output_path: str = ""
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, object]:
        """Serialize job for API output."""
        return asdict(self)


class JobStore:
    """Keep local job state for the API process."""

    def __init__(self) -> None:
        """Create an empty job registry."""
        self.jobs: dict[str, Job] = {}

    def create(self) -> Job:
        """Create a queued job."""
        job = Job(id=str(uuid4()))
        self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return one job or None."""
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Mark a queued/running job as cancelled."""
        job = self.get(job_id)
        if job and job.status in {"queued", "running"}:
            job.status = "cancelled"
        return job
