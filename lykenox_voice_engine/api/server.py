"""Local FastAPI service for LYKENOX Voice Engine."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from lykenox_voice_engine.api.jobs import JobRegistry
from lykenox_voice_engine.api.schemas import JobResponse, SynthesizeMidiRequest, SynthesizeRequest
from lykenox_voice_engine.config.settings import load_settings
from lykenox_voice_engine.core.profile_manager import ProfileManager
from lykenox_voice_engine.engines.nnsvs_engine import NnsvsEngine
from lykenox_voice_engine.models.note import NoteSequence


def create_app(root: Path | None = None) -> FastAPI:
    """Create the local API application."""

    app_root = root or Path(__file__).resolve().parents[2]
    settings = load_settings(app_root)
    profiles = ProfileManager(settings.profiles_dir)
    engine = NnsvsEngine(app_root)
    jobs = JobRegistry()
    app = FastAPI(title="LYKENOX Voice Engine", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        backend = engine.check_available()
        return {
            "ok": True,
            "device": settings.device,
            "backend": "nnsvs",
            "backend_available": bool(backend.get("available")),
            "training_device": "cpu",
            "runtime": backend,
        }

    @app.get("/profiles")
    def list_profiles() -> list[dict[str, object]]:
        return [profile.__dict__ for profile in profiles.list_profiles()]

    @app.get("/profiles/{profile_id}")
    def get_profile(profile_id: str) -> dict[str, object]:
        try:
            return profiles.get_profile(profile_id).__dict__
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/synthesize", response_model=JobResponse)
    def synthesize(payload: SynthesizeRequest) -> JobResponse:
        NoteSequence.from_dict({"tempo": payload.tempo, "notes": [n.model_dump() for n in payload.notes]})
        job = jobs.create()
        try:
            output = engine.synthesize(payload.profile, payload.lyrics, [], payload.tempo)
            job.status = "completed"
            job.output_path = str(output)
        except RuntimeError as exc:
            job.status = "failed"
            job.error = str(exc)
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

    @app.post("/synthesize-midi", response_model=JobResponse)
    def synthesize_midi(payload: SynthesizeMidiRequest) -> JobResponse:
        job = jobs.create()
        job.status = "failed"
        job.error = "MIDI adapter pendiente: primero validar MusicXML/UST a HTS full-context labels."
        return JobResponse(job_id=job.id, status=job.status, error=job.error)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

    @app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str) -> JobResponse:
        engine.cancel(job_id)
        job = jobs.cancel(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

    return app


app = create_app()
