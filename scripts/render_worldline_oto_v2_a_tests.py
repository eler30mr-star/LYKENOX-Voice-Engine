"""Render a.wav WORLDLINE-R tests with candidate oto_worldline.ini only."""

from __future__ import annotations

import ctypes
import json
import math
import sys
import time
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.oto import parse_oto
from lykenox_voice_engine.engines.openutau_phrase_adapter import OpenUtauPhraseAdapter
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine
from lykenox_voice_engine.models.notes import NoteEvent


def main() -> None:
    """Render MIDI 60/62/64 with the candidate OTO for a.wav."""

    voicebank = ROOT / "profiles" / "lykenox" / "voicebank"
    wav_dir = voicebank / "wav"
    oto = parse_oto(voicebank / "oto_worldline.ini")
    output_dir = ROOT / "outputs" / "worldline_pitch_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    worldline = OpenUtauWorldlineEngine(ROOT)
    cases = {
        60: output_dir / "a_midi60_oto_v2.wav",
        62: output_dir / "a_midi62_oto_v2.wav",
        64: output_dir / "a_midi64_oto_v2.wav",
    }
    started = time.perf_counter()
    results = []
    for midi, path in cases.items():
        notes = [NoteEvent("a", midi, 0.0, 1.5)]
        report = worldline.render_to_path(wav_dir, oto, lambda lyric: [lyric], notes, 120, path)
        results.append(validate(path, notes[0], oto["a"], report.render_time_sec))
    print(
        json.dumps(
            {
                "oto_path": str(voicebank / "oto_worldline.ini"),
                "render_time_sec": round(time.perf_counter() - started, 3),
                "all_pitch_ok": all(item["pitch_ok"] for item in results),
                "results": results,
            },
            indent=2,
        )
    )


def validate(path: Path, note: NoteEvent, entry, render_time_sec: float) -> dict[str, object]:
    """Validate F0 over the stable central output region."""

    sample_rate, samples = read_wav(path)
    f0 = estimate_f0(samples, sample_rate)
    stable_start_sec = max(0.2, (entry.preutterance + entry.consonant + 250.0) / 1000.0)
    stable_end_sec = min(len(samples) / sample_rate - 0.2, stable_start_sec + 0.8)
    voiced = [value for value in f0[int(stable_start_sec * 100) : int(stable_end_sec * 100)] if value > 0]
    measured = sum(voiced) / len(voiced) if voiced else 0.0
    target = midi_to_hz(note.midi)
    cents = 1200 * math.log2(measured / target) if measured > 0 else None
    deltas = [abs(samples[index] - samples[index - 1]) for index in range(1, len(samples))]
    useful_ms = wav_duration_ms(path) - entry.offset - entry.cutoff
    return {
        "path": str(path),
        "midi": note.midi,
        "target_hz": round(target, 2),
        "measured_hz": round(measured, 2),
        "error_cents": round(cents, 1) if cents is not None else None,
        "pitch_ok": bool(cents is not None and abs(cents) <= 30),
        "duration_sec": round(len(samples) / sample_rate, 3),
        "render_time_sec": round(render_time_sec, 3),
        "stable_window_sec": [round(stable_start_sec, 3), round(stable_end_sec, 3)],
        "voiced_frames": len(voiced),
        "peak": round(max((abs(sample) for sample in samples), default=0.0), 4),
        "click_candidates_gt_0_8": sum(1 for delta in deltas if delta > 0.8),
        "max_sample_delta": round(max(deltas, default=0.0), 4),
        "oto": {
            "offset": entry.offset,
            "consonant": entry.consonant,
            "cutoff": entry.cutoff,
            "preutterance": entry.preutterance,
            "overlap": entry.overlap,
            "useful_ms": round(useful_ms, 3),
        },
    }


def read_wav(path: Path) -> tuple[int, list[float]]:
    """Read PCM16 WAV as normalized floats."""

    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    values = array("h")
    values.frombytes(raw)
    return sample_rate, [value / 32768.0 for value in values]


def wav_duration_ms(path: Path) -> float:
    """Return WAV duration in milliseconds."""

    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / reader.getframerate() * 1000.0


def estimate_f0(samples: list[float], sample_rate: int) -> list[float]:
    """Estimate F0 with WORLDLINE-R's F0 export."""

    worldline = OpenUtauWorldlineEngine(ROOT)
    dll = worldline._load()
    sample_array = (ctypes.c_float * len(samples))(*samples)
    f0_ptr = ctypes.POINTER(ctypes.c_double)()
    length = dll.F0(sample_array, len(samples), sample_rate, 10.0, 0, ctypes.byref(f0_ptr))
    return [float(f0_ptr[index]) for index in range(length)] if length > 0 else []


def midi_to_hz(midi: int) -> float:
    """Convert MIDI to Hz."""

    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


if __name__ == "__main__":
    main()
