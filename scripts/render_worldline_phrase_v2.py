"""Render and validate the WORLDLINE-R real v2 phrase."""

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
from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine
from lykenox_voice_engine.models.notes import NoteEvent


def main() -> None:
    """Render bai/la/con/mi/go and print validation JSON."""

    notes = [
        NoteEvent("bai", 60, 0.0, 0.5),
        NoteEvent("la", 62, 0.5, 0.5),
        NoteEvent("con", 64, 1.0, 0.5),
        NoteEvent("mi", 62, 1.5, 0.5),
        NoteEvent("go", 60, 2.0, 0.75),
    ]
    out = ROOT / "outputs" / "comparison" / "vocal_worldline_real_v2.wav"
    started = time.perf_counter()
    UtauSampleEngine(ROOT).synthesize_to_path(
        "lykenox",
        "baila conmigo",
        notes,
        120,
        out,
        renderer="worldline_real",
    )
    render_time = time.perf_counter() - started
    print(json.dumps(validate(out, notes, render_time), indent=2))


def validate(path: Path, notes: list[NoteEvent], render_time_sec: float) -> dict[str, object]:
    """Validate phrase F0 in stable request windows."""

    sample_rate, samples = read_wav(path)
    f0 = estimate_f0(samples, sample_rate)
    metadata = request_metadata(notes)
    per_note = []
    for index, note in enumerate(notes):
        meta = metadata[index]
        stable_start = (
            meta["phrase_pos_ms"]
            + meta["preutterance"]
            + meta["consonant"]
            + 120.0
        ) / 1000.0
        stable_end = min(len(samples) / sample_rate - 0.05, stable_start + 0.22)
        voiced = [value for value in f0[int(stable_start * 100) : int(stable_end * 100)] if value > 0]
        measured = sum(voiced) / len(voiced) if voiced else 0.0
        target = midi_to_hz(note.midi)
        cents = 1200 * math.log2(measured / target) if measured > 0 else None
        per_note.append(
            {
                "alias": note.lyric,
                "target_midi": note.midi,
                "target_hz": round(target, 2),
                "measured_hz": round(measured, 2),
                "error_cents": round(cents, 1) if cents is not None else None,
                "pitch_ok": bool(cents is not None and abs(cents) <= 30),
                "start_time": note.start,
                "duration": note.duration,
                "stable_window_sec": [round(stable_start, 3), round(stable_end, 3)],
                **meta,
            }
        )
    deltas = [abs(samples[index] - samples[index - 1]) for index in range(1, len(samples))]
    return {
        "path": str(path),
        "render_time_sec": round(render_time_sec, 3),
        "duration_sec": round(len(samples) / sample_rate, 3),
        "peak": round(max((abs(sample) for sample in samples), default=0.0), 4),
        "click_candidates_gt_0_8": sum(1 for delta in deltas if delta > 0.8),
        "max_sample_delta": round(max(deltas, default=0.0), 4),
        "all_pitch_ok": all(item["pitch_ok"] for item in per_note),
        "per_note": per_note,
    }


def request_metadata(notes: list[NoteEvent]) -> list[dict[str, float]]:
    """Return OpenUtau request metadata for the phrase."""

    voicebank = ROOT / "profiles" / "lykenox" / "voicebank"
    oto = parse_oto(voicebank / "oto.ini")
    adapter = OpenUtauPhraseAdapter(
        voicebank / "wav",
        oto,
        lambda lyric: [lyric],
        120,
    )
    phrase = adapter.build_phrase(notes)
    return [
        {
            "preutterance": round(request.phone.preutter_ms, 3),
            "overlap": round(request.phone.overlap_ms, 3),
            "consonant": round(request.consonant, 3),
            "skipOver": round(request.skip_over, 3),
            "cutoff": round(request.cutoff, 3),
            "durRequired": round(request.dur_required, 3),
            "phrase_pos_ms": round(request.pos_ms, 3),
            "length_ms": round(request.length_ms, 3),
        }
        for request in phrase.requests
    ]


def read_wav(path: Path) -> tuple[int, list[float]]:
    """Read PCM16 WAV as normalized floats."""

    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    values = array("h")
    values.frombytes(raw)
    return sample_rate, [value / 32768.0 for value in values]


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
