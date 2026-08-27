"""Local FastAPI service for LYKENOX Voice Engine."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from lykenox_voice_engine.api.jobs import JobRegistry
from lykenox_voice_engine.api.schemas import (
    JobResponse,
    SingRequest,
    SpeakRequest,
    SynthesizeMidiRequest,
    SynthesizeRequest,
)
from lykenox_voice_engine.config.settings import load_settings
from lykenox_voice_engine.core.midi import parse_midi
from lykenox_voice_engine.core.neural_voice_contract import SingingSynthesisRequest, SpeechSynthesisRequest
from lykenox_voice_engine.core.profile_manager import ProfileManager
from lykenox_voice_engine.engines.identity_voice_engine import (
    IdentityModelUnavailableError,
    IdentityVoiceEngine,
)
from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.models.job import JobStatus
from lykenox_voice_engine.models.note import NoteSequence
from lykenox_voice_engine.models.notes import NoteEvent


def create_app(root: Path | None = None) -> FastAPI:
    """Create the local API application."""

    app_root = root or Path(__file__).resolve().parents[2]
    settings = load_settings(app_root)
    profiles = ProfileManager(settings.profiles_dir)
    engine = UtauSampleEngine(app_root)
    identity_engine = IdentityVoiceEngine(app_root)
    jobs = JobRegistry()
    app = FastAPI(title="LYKENOX Voice Engine", version="0.3.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        backend = engine.check_available()
        return {
            "ok": True,
            "device": "cpu",
            "backend": "identity_voice_target",
            "backend_available": bool(backend.get("available")),
            "identity_model": identity_engine.status(),
            "legacy_backend": "utau_worldline_fallback",
            "voicebank_available": bool(backend.get("voicebank_available")),
            "voicebank_coverage": backend.get("voicebank_coverage", 0.0),
            "renderer_available": bool(backend.get("renderer_available")),
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
        """Render the legacy sample-based singing endpoint."""

        return _sample_sing(payload, "vocal.wav")

    @app.post("/sing-sample", response_model=JobResponse)
    def sing_sample(payload: SynthesizeRequest) -> JobResponse:
        """Render direct local singing from the LYKENOX voicebank."""

        return _sample_sing(payload, "singing_sample.wav")

    def _sample_sing(payload: SynthesizeRequest, filename: str) -> JobResponse:
        """Render sample-based singing and return a job response."""

        sequence = NoteSequence.from_dict(
            {"tempo": payload.tempo, "notes": [note.model_dump() for note in payload.notes]}
        )
        notes = [NoteEvent(note.lyric, note.midi, note.start, note.duration) for note in sequence.notes]
        job = jobs.create()
        job.status = JobStatus.RUNNING
        output_path = app_root / "outputs" / job.id / filename
        try:
            engine.synthesize_to_path(payload.profile, payload.lyrics, notes, payload.tempo, output_path)
            job.status = JobStatus.COMPLETED
            job.output_path = str(output_path)
        except RuntimeError as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

    @app.post("/speak", response_model=JobResponse)
    def speak(payload: SpeakRequest) -> JobResponse:
        job = jobs.create()
        job.status = JobStatus.RUNNING
        output_path = app_root / "outputs" / job.id / "speech.wav"
        try:
            identity_engine.synthesize_speech(
                SpeechSynthesisRequest(
                    profile=payload.profile,
                    text=payload.text,
                    language=payload.language,
                    speaking_rate=payload.speaking_rate,
                ),
                output_path,
            )
            job.status = JobStatus.COMPLETED
            job.output_path = str(output_path)
        except IdentityModelUnavailableError as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

    @app.post("/sing", response_model=JobResponse)
    def sing(payload: SingRequest) -> JobResponse:
        notes = [NoteEvent(note.lyric, note.midi, note.start, note.duration) for note in payload.notes]
        job = jobs.create()
        job.status = JobStatus.RUNNING
        output_path = app_root / "outputs" / job.id / "singing.wav"
        try:
            identity_engine.synthesize_singing(
                SingingSynthesisRequest(
                    profile=payload.profile,
                    lyrics=payload.lyrics,
                    notes=notes,
                    tempo=payload.tempo,
                    language=payload.language,
                ),
                output_path,
            )
            job.status = JobStatus.COMPLETED
            job.output_path = str(output_path)
        except IdentityModelUnavailableError as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

    @app.post("/synthesize-midi", response_model=JobResponse)
    def synthesize_midi(payload: SynthesizeMidiRequest) -> JobResponse:
        job = jobs.create()
        job.status = JobStatus.RUNNING
        output_path = app_root / "outputs" / job.id / "vocal.wav"
        try:
            parsed = parse_midi(Path(payload.midi_path), external_lyrics=payload.lyrics.split())
            engine.synthesize_to_path(payload.profile, payload.lyrics, list(parsed.notes), parsed.tempo, output_path)
            job.status = JobStatus.COMPLETED
            job.output_path = str(output_path)
        except (RuntimeError, ValueError, OSError) as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
        return JobResponse(job_id=job.id, status=job.status, output_path=job.output_path, error=job.error)

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
