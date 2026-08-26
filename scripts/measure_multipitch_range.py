"""Measure current LYKENOX voicebank F0 and design multipitch centers."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.multipitch import design_pitch_centers, report_to_dict
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine


def main() -> None:
    """Measure every usable WAV in the current monopitch voicebank."""

    voicebank = ROOT / "profiles" / "lykenox" / "voicebank"
    wav_dir = voicebank / "wav"
    engine = OpenUtauWorldlineEngine(ROOT)
    measurements = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        if wav_path.name.startswith(".") or wav_path.stat().st_size <= 44:
            continue
        samples = _read_wav_mono_float(wav_path)
        mean_f0 = engine.estimate_mean_f0(samples)
        if mean_f0 <= 0:
            continue
        measurements.append(
            {
                "alias": wav_path.stem.lower(),
                "file": wav_path.name,
                "duration_sec": round(_duration(wav_path), 3),
                "mean_f0_hz": round(mean_f0, 2),
            }
        )

    report = design_pitch_centers([item["mean_f0_hz"] for item in measurements])
    payload = {
        "voicebank": str(voicebank),
        "measured_from": str(wav_dir),
        "summary": report_to_dict(report),
        "measurements": measurements,
    }
    out_path = voicebank / "multipitch_pitch_report.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"report={out_path}")


def _read_wav_mono_float(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        raw = reader.readframes(reader.getnframes())
    if width != 2:
        raise ValueError(f"Solo PCM16 soportado: {path}")
    values = []
    for index in range(0, len(raw), width * channels):
        sample = int.from_bytes(raw[index : index + 2], "little", signed=True)
        values.append(sample / 32768.0)
    return values


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / float(reader.getframerate())


if __name__ == "__main__":
    main()
