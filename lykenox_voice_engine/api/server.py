"""Local-only FastAPI server for LYKENOX Voice Engine."""

from fastapi import FastAPI, HTTPException

from lykenox_voice_engine.api.jobs import JobStore
from lykenox_voice_engine.api.schemas import JobResponse, SynthesizeRequest
from lykenox_voice_engine.config.settings import load_settings
from lykenox_voice_engine.core.profile_manager import ProfileManager

app = FastAPI(title="LYKENOX Voice Engine")
jobs = JobStore()
profiles = ProfileManager()


@app.get("/health")
def health() -> dict[str, object]:
    """Return local API status."""
    settings = load_settings()
    return {"status": "ok", "host": settings["api_host"], "backend": "audit-only"}


@app.get("/profiles")
def list_profiles() -> list[dict[str, object]]:
    """List voice profiles."""
    return [profile.to_dict() for profile in profiles.list_profiles()]


@app.get("/profiles/{profile_id}")
def get_profile(profile_id: str) -> dict[str, object]:
    """Return one profile."""
    try:
        return profiles.get_profile(profile_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/synthesize", response_model=JobResponse)
def synthesize(request: SynthesizeRequest) -> JobResponse:
    """Queue a score-to-singing request without requiring source vocal audio."""
    job = jobs.create()
    job.status = "failed"
    job.error = "No hay backend SVS real seleccionado todavía."
    return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path)


@app.post("/synthesize-midi", response_model=JobResponse)
def synthesize_midi() -> JobResponse:
    """Placeholder for future MIDI import."""
    job = jobs.create()
    job.status = "failed"
    job.error = "Importación MIDI pendiente de backend seleccionado."
    return JobResponse(job_id=job.id, status=job.status)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    """Return job state."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return job.to_dict()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, object]:
    """Cancel a queued or running job."""
    job = jobs.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return job.to_dict()


def main() -> None:
    """Run the local API on 127.0.0.1 only."""
    import uvicorn

    settings = load_settings()
    uvicorn.run(app, host=settings["api_host"], port=int(settings["api_port"]))


if __name__ == "__main__":
    main()
