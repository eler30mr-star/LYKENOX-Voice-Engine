"""Validate the WORLDLINE-R real microtest render."""

from __future__ import annotations

import ctypes
import json
import math
import sys
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine


def main() -> None:
    """Print technical validation details for vocal_worldline_real.wav."""

    root = ROOT
    out = root / "outputs" / "comparison" / "vocal_worldline_real.wav"
    with wave.open(str(out), "rb") as reader:
        sample_rate = reader.getframerate()
        frames = reader.getnframes()
        width = reader.getsampwidth()
        channels = reader.getnchannels()
        raw = reader.readframes(frames)

    values = array("h")
    values.frombytes(raw)
    samples = [value / 32768.0 for value in values]

    engine = OpenUtauWorldlineEngine(root)
    dll = engine._load()
    sample_array = (ctypes.c_float * len(samples))(*samples)
    f0_ptr = ctypes.POINTER(ctypes.c_double)()
    f0_length = dll.F0(
        sample_array,
        len(samples),
        sample_rate,
        10.0,
        0,
        ctypes.byref(f0_ptr),
    )
    f0 = [float(f0_ptr[index]) for index in range(f0_length)] if f0_length > 0 else []
    notes = [
        ("bai", 60, 0.0, 0.5),
        ("la", 62, 0.5, 0.5),
        ("con", 64, 1.0, 0.5),
        ("mi", 62, 1.5, 0.5),
        ("go", 60, 2.0, 0.75),
    ]

    per_note = []
    for lyric, midi, start, duration in notes:
        start_frame = int(start * 100)
        end_frame = int((start + duration) * 100)
        voiced = [value for value in f0[start_frame:end_frame] if value > 0]
        mean_f0 = sum(voiced) / len(voiced) if voiced else 0.0
        expected = _midi_to_hz(midi)
        cents = 1200 * math.log2(mean_f0 / expected) if mean_f0 > 0 else None
        per_note.append(
            {
                "lyric": lyric,
                "midi": midi,
                "expected_hz": round(expected, 2),
                "mean_f0_hz": round(mean_f0, 2),
                "cents_error": round(cents, 1) if cents is not None else None,
                "pitch_ok": bool(cents is not None and abs(cents) < 250),
            }
        )

    peak = max((abs(sample) for sample in samples), default=0.0)
    deltas = [abs(samples[index] - samples[index - 1]) for index in range(1, len(samples))]
    old_outputs = {
        path.name: {"bytes": path.stat().st_size}
        for path in sorted((root / "outputs" / "comparison").glob("*.wav"))
    }
    summary = {
        "path": str(out),
        "exists": out.exists(),
        "bytes": out.stat().st_size if out.exists() else 0,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": width,
        "duration_sec": round(frames / sample_rate, 3),
        "peak": round(peak, 4),
        "f0_frames": len(f0),
        "per_note": per_note,
        "all_pitch_ok": all(item["pitch_ok"] for item in per_note),
        "click_candidates_gt_0_8": sum(1 for delta in deltas if delta > 0.8),
        "max_sample_delta": round(max(deltas, default=0.0), 4),
        "old_outputs": old_outputs,
    }
    print(json.dumps(summary, indent=2))


def _midi_to_hz(midi: int) -> float:
    """Convert MIDI note number to frequency in Hz."""

    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


if __name__ == "__main__":
    main()
