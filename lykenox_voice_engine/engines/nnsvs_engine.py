"""NNSVS/ENUNU CPU microtest backend boundary."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from lykenox_voice_engine.engines.singing_engine import SingingVoiceEngine
from lykenox_voice_engine.models.notes import NoteEvent


class NnsvsEngine(SingingVoiceEngine):
    """Validate and run a safe NNSVS CPU microtest when dependencies exist."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.env_python = self.root / "tools" / "nnsvs_env" / ".venv" / "Scripts" / "python.exe"
        self.microtest_dir = self.root / "datasets" / "lykenox" / "microtest"
        self.output_dir = self.root / "outputs" / "microtest"
        self._cancelled = False

    def check_available(self) -> dict[str, Any]:
        """Return real NNSVS runtime state from the isolated backend env."""

        if not self.env_python.exists():
            return {"available": False, "backend": "nnsvs", "reason": "tools/nnsvs_env .venv no existe"}
        script = self.root / "tools" / "nnsvs_env" / "check_nnsvs_runtime.py"
        result = subprocess.run(
            [str(self.env_python), str(script)],
            cwd=str(self.root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError:
            payload = {"available": False, "error": result.stdout[-800:]}
        payload["backend"] = "nnsvs"
        payload["returncode"] = result.returncode
        if result.stderr:
            payload["stderr"] = result.stderr[-1200:]
        return payload

    def prepare_dataset(self, profile: str) -> dict[str, Any]:
        """Copy a tiny authorized sample set and write the required score audit files."""

        raw_dir = self.root / "datasets" / profile / "raw"
        self.microtest_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        total_seconds = 0.0
        for source in sorted(raw_dir.glob("*")):
            if source.suffix.lower() not in {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus"}:
                continue
            target = self.microtest_dir / "raw" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            copied.append(str(target))
            total_seconds += 15.0
            if total_seconds >= 60.0:
                break
        score = self._micro_score()
        (self.microtest_dir / "score").mkdir(parents=True, exist_ok=True)
        (self.microtest_dir / "score" / "baila_conmigo.json").write_text(
            json.dumps(score, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "ok": bool(copied),
            "profile": profile,
            "copied_files": copied,
            "score": str(self.microtest_dir / "score" / "baila_conmigo.json"),
            "status": "needs_hts_full_context_labels",
            "reason": "NNSVS requiere MusicXML/UST a HTS full-context labels alineados con WAV.",
        }

    def train(self, profile: str) -> dict[str, Any]:
        """Run a real NNSVS microtraining only when the backend import succeeds."""

        status = self.check_available()
        if not status.get("available"):
            return self._blocked_training_result(status)
        return {
            "ok": False,
            "profile": profile,
            "reason": "Runtime disponible, pero receta NNSVS española/HTS labels aún no configurada.",
            "preprocessing": "blocked",
        }

    def resume_training(self, profile: str, checkpoint: str) -> dict[str, Any]:
        """Resume only if the checkpoint exists and NNSVS is importable."""

        if not Path(checkpoint).exists():
            return {"ok": False, "profile": profile, "checkpoint": checkpoint, "reason": "checkpoint no existe"}
        status = self.check_available()
        return {"ok": False, "profile": profile, "checkpoint": checkpoint, "runtime": status}

    def synthesize(self, profile: str, lyrics: str, notes: list[NoteEvent], tempo: int) -> Path:
        """Generate vocal.wav only after a valid NNSVS packed model exists."""

        model_dir = self.root / "profiles" / profile / "model" / "nnsvs_packed_model"
        checkpoint = model_dir / "acoustic_model.pth"
        if not checkpoint.exists():
            raise RuntimeError("No existe checkpoint/modelo NNSVS válido; no se genera audio falso.")
        raise RuntimeError("NNSVS runtime existe, pero la síntesis real aún requiere adaptador MusicXML/UST.")

    def cancel(self, job_id: str | None = None) -> None:
        """Record cancellation for cooperative backend jobs."""

        self._cancelled = True

    def get_model_info(self) -> dict[str, Any]:
        """Return model/checkpoint availability for the LYKENOX profile."""

        model_dir = self.root / "profiles" / "lykenox" / "model" / "nnsvs_packed_model"
        return {
            "backend": "nnsvs",
            "model_dir": str(model_dir),
            "checkpoint_exists": (model_dir / "acoustic_model.pth").exists(),
            "vocoder": "WORLD planned for microtest",
        }

    def _blocked_training_result(self, status: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        return {
            "ok": False,
            "preprocessing": "not_run",
            "forward": "not_run",
            "backward": "not_run",
            "optimizer_step": "not_run",
            "checkpoint": None,
            "time_per_step": None,
            "ram_peak_mb": None,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "reason": "NNSVS no está disponible en tools/nnsvs_env.",
            "runtime": status,
        }

    def _micro_score(self) -> dict[str, Any]:
        return {
            "tempo": 120,
            "lyrics": "baila conmigo",
            "notes": [
                {"syllable": "bai", "midi": 60, "start": 0.0, "duration": 0.5},
                {"syllable": "la", "midi": 62, "start": 0.5, "duration": 0.5},
                {"syllable": "con", "midi": 64, "start": 1.0, "duration": 0.5},
                {"syllable": "mi", "midi": 62, "start": 1.5, "duration": 0.5},
                {"syllable": "go", "midi": 60, "start": 2.0, "duration": 0.75},
            ],
            "spanish_phonemes_required": ["b", "a", "i", "l", "k", "o", "n", "m", "g"],
            "nnsvs_required_label_format": "HTS full-context labels generated from MusicXML or UST",
        }
