"""UTAU-style Spanish sample voicebank backend."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from lykenox_voice_engine.core.voicebank import VoicebankManager
from lykenox_voice_engine.engines.singing_engine import SingingVoiceEngine
from lykenox_voice_engine.models.notes import NoteEvent


class UtauSampleEngine(SingingVoiceEngine):
    """Render singing directly from score and Spanish Lite voicebank samples."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self._cancelled = False

    def check_available(self) -> dict[str, Any]:
        """Return renderer and voicebank availability."""

        manager = VoicebankManager(self.root)
        from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine

        worldline = OpenUtauWorldlineEngine(self.root).health_check()
        validation = manager.validate_voicebank()
        microtest = manager.microtest_status()
        return {
            "available": True,
            "backend": "utau_sample",
            "renderer_available": True,
            "renderer": "internal_concat_pcm",
            "voicebank_available": validation["voicebank_available"],
            "voicebank_coverage": validation["voicebank_coverage"],
            "microtest": microtest,
            "worldline_r": worldline,
            "nnsvs": "experimental_not_recommended",
        }

    def prepare_dataset(self, profile: str) -> dict[str, Any]:
        """Prepare the voicebank directories for profile recording."""

        manager = VoicebankManager(self.root, profile)
        manager.raw_dir.mkdir(parents=True, exist_ok=True)
        manager.wav_dir.mkdir(parents=True, exist_ok=True)
        reclist = manager.load_reclist()
        return {
            "ok": True,
            "profile": profile,
            "raw_dir": str(manager.raw_dir),
            "voicebank_dir": str(manager.voicebank_dir),
            "reclist_count": len(reclist.aliases),
            "format": "48k mono PCM 16-bit WAV",
        }

    def build_voicebank(self, profile: str = "lykenox") -> dict[str, Any]:
        """Build oto.ini and copy accepted WAVs into the profile voicebank."""

        return VoicebankManager(self.root, profile).build_voicebank()

    def validate_voicebank(self, profile: str = "lykenox") -> dict[str, Any]:
        """Validate Spanish Lite voicebank readiness."""

        return VoicebankManager(self.root, profile).validate_voicebank()

    def train(self, profile: str) -> dict[str, Any]:
        """Disable neural training for the sample-based backend."""

        return {
            "ok": False,
            "profile": profile,
            "backend": "utau_sample",
            "reason": "Este backend no entrena redes; usa grabaciones WAV + oto.ini.",
        }

    def resume_training(self, profile: str, checkpoint: str) -> dict[str, Any]:
        """Disable checkpoint resume because this backend is sample-based."""

        return {
            "ok": False,
            "profile": profile,
            "checkpoint": checkpoint,
            "backend": "utau_sample",
            "reason": "No hay checkpoints neurales en la ruta sample-based.",
        }

    def synthesize(self, profile: str, lyrics: str, notes: list[NoteEvent], tempo: int) -> Path:
        """Generate vocal.wav from voicebank samples and score notes."""

        output_path = self.root / "outputs" / str(uuid.uuid4()) / "vocal.wav"
        return self.synthesize_to_path(profile, lyrics, notes, tempo, output_path)

    def synthesize_to_path(
        self,
        profile: str,
        lyrics: str,
        notes: list[NoteEvent],
        tempo: int,
        output_path: Path,
        renderer: str = "internal"
    ) -> Path:
        """Generate vocal.wav at a caller-owned path."""

        manager = VoicebankManager(self.root, profile)
        manager.render_to_path(lyrics, notes, tempo, output_path, renderer_type=renderer)
        return output_path

    def compile_worldline(self) -> bool:
        """Attempt to compile the local UTAU bridge."""
        from lykenox_voice_engine.core.resampler_interface import UtauClassicRenderer
        return UtauClassicRenderer(self.root).compile_worldline()

    def worldline_health(self) -> dict[str, Any]:
        """Return OpenUtau WORLDLINE-R native wrapper health."""

        from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine

        return OpenUtauWorldlineEngine(self.root).health_check()

    def coverage_for(self, profile: str, lyrics: str, notes: list[NoteEvent]) -> dict[str, Any]:
        """Return coverage details for a synthesis request."""

        report = VoicebankManager(self.root, profile).coverage_for(lyrics, notes)
        return {
            "required": list(report.required),
            "available": list(report.available),
            "missing": list(report.missing),
            "coverage": report.coverage,
        }

    def cancel(self, job_id: str | None = None) -> None:
        """Mark current cooperative render as cancelled."""

        del job_id
        self._cancelled = True

    def get_model_info(self) -> dict[str, Any]:
        """Return sample-backend model metadata."""

        manager = VoicebankManager(self.root)
        validation = manager.validate_voicebank()
        return {
            "backend": "utau_sample",
            "model": "LYKENOX Spanish Lite voicebank",
            "voicebank_dir": str(manager.voicebank_dir),
            "renderer": "internal_concat_pcm",
            "training": "not_used",
            "validation": validation,
        }
