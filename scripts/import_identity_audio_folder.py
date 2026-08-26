"""Import existing voice recordings into the LYKENOX identity dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.identity_dataset import IdentityDatasetService, IdentityPrompt  # noqa: E402


def main() -> None:
    """Import a folder of existing audio as speech/singing identity takes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    service = IdentityDatasetService(ROOT)
    service.ensure_structure()
    imported = []
    for source in sorted(args.source_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus"}:
            continue
        mode = _classify_mode(source.name)
        prompt = IdentityPrompt(
            id=f"import-{source.stem.lower().replace(' ', '-')}",
            mode=mode,
            text=_text_from_name(source.stem),
            melody_hint="audio importado" if mode == "singing" else None,
        )
        raw_wav = service.raw_path(mode, prompt.id)
        if source.suffix.lower() == ".wav":
            shutil.copy2(source, raw_wav)
        else:
            _convert_to_wav(source, raw_wav)
        metadata = service.register_take(mode, prompt, raw_wav)
        imported.append(
            {
                "source": str(source),
                "mode": mode,
                "status": metadata.status,
                "reason": metadata.reason,
                "duration_sec": metadata.duration_sec,
                "f0_hz": metadata.measured_f0_hz,
                "wav_path": metadata.wav_path,
            }
        )
    report = {"source_dir": str(args.source_dir), "imported": imported, "summary": service.summary()}
    out_path = service.metadata_dir / "last_import_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _classify_mode(name: str) -> str:
    text = name.lower()
    singing_markers = ("canto", "vocales", "notas", "registro")
    return "singing" if any(marker in text for marker in singing_markers) else "speech"


def _text_from_name(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip()


def _convert_to_wav(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-ar", "48000", "-ac", "1", "-sample_fmt", "s16", str(target)],
        capture_output=True,
        check=True,
        timeout=120,
    )


if __name__ == "__main__":
    main()
