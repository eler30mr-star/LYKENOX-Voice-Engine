"""Render and validate minimal WORLDLINE-R pitch tests."""

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

from lykenox_voice_engine.engines.utau_engine import UtauSampleEngine
from lykenox_voice_engine.core.oto import parse_oto
from lykenox_voice_engine.engines.openutau_phrase_adapter import OpenUtauPhraseAdapter
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine
from lykenox_voice_engine.models.notes import NoteEvent


def main() -> None:
    """Render A-D pitch tests and print validation JSON."""

    output_dir = ROOT / "outputs" / "worldline_pitch_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = UtauSampleEngine(ROOT)
    cases = {
        "a_midi_60": [NoteEvent("a", 60, 0.0, 1.5)],
        "a_midi_62": [NoteEvent("a", 62, 0.0, 1.5)],
        "a_midi_64": [NoteEvent("a", 64, 0.0, 1.5)],
        "a_sequence": [
            NoteEvent("a", 60, 0.0, 1.5),
            NoteEvent("a", 62, 1.5, 1.5),
            NoteEvent("a", 64, 3.0, 1.5),
            NoteEvent("a", 62, 4.5, 1.5),
            NoteEvent("a", 60, 6.0, 1.5),
        ],
    }
    results = {}
    started = time.perf_counter()
    for name, notes in cases.items():
        path = output_dir / f"{name}.wav"
        engine.synthesize_to_path("lykenox", " ".join(note.lyric for note in notes), notes, 120, path, renderer="worldline_real")
        results[name] = validate(path, notes, request_metadata(notes))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "render_time_sec": round(time.perf_counter() - started, 3),
                "all_ok": all(result["all_pitch_ok"] for result in results.values()),
                "results": results,
            },
            indent=2,
        )
    )


def validate(path: Path, notes: list[NoteEvent], metadata: list[dict[str, float]]) -> dict[str, object]:
    """Validate stable F0 windows for one rendered WAV."""

    sample_rate, samples = _read_wav(path)
    f0 = _f0(samples, sample_rate)
    per_note = []
    for index, note in enumerate(notes):
        request_meta = metadata[index]
        stable_start = (
            request_meta["phrase_pos_ms"]
            + request_meta["preutterance"]
            + request_meta["consonant"]
            + 250.0
        ) / 1000.0
        stable_end = min(
            len(samples) / sample_rate - 0.1,
            stable_start + min(0.8, max(0.2, note.duration * 0.55)),
        )
        start_frame = max(0, int(stable_start * 100))
        end_frame = min(len(f0), int(stable_end * 100))
        voiced = [value for value in f0[start_frame:end_frame] if value > 0]
        measured = sum(voiced) / len(voiced) if voiced else 0.0
        target = _midi_to_hz(note.midi)
        cents = 1200 * math.log2(measured / target) if measured > 0 else None
        per_note.append(
            {
                "alias": note.lyric,
                "target_midi": note.midi,
                "target_hz": round(target, 2),
                "measured_f0_hz": round(measured, 2),
                "error_cents": round(cents, 1) if cents is not None else None,
                "pitch_ok": bool(cents is not None and abs(cents) <= 30),
                "start_time": note.start,
                "duration": note.duration,
                "stable_window_sec": [round(stable_start, 3), round(stable_end, 3)],
                **request_meta,
            }
        )
    deltas = [abs(samples[index] - samples[index - 1]) for index in range(1, len(samples))]
    return {
        "path": str(path),
        "duration_sec": round(len(samples) / sample_rate, 3),
        "peak": round(max((abs(sample) for sample in samples), default=0.0), 4),
        "click_candidates_gt_0_8": sum(1 for delta in deltas if delta > 0.8),
        "max_sample_delta": round(max(deltas, default=0.0), 4),
        "all_pitch_ok": all(item["pitch_ok"] for item in per_note),
        "per_note": per_note,
    }


def request_metadata(notes: list[NoteEvent]) -> list[dict[str, float]]:
    """Return OpenUtau request timing metadata for each note."""

    oto = parse_oto(ROOT / "profiles" / "lykenox" / "voicebank" / "oto.ini")
    adapter = OpenUtauPhraseAdapter(
        ROOT / "profiles" / "lykenox" / "voicebank" / "wav",
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


def _read_wav(path: Path) -> tuple[int, list[float]]:
    """Read a mono PCM16 WAV."""

    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    values = array("h")
    values.frombytes(raw)
    return sample_rate, [value / 32768.0 for value in values]


def _f0(samples: list[float], sample_rate: int) -> list[float]:
    """Estimate F0 with the official WORLDLINE-R F0 export."""

    worldline = OpenUtauWorldlineEngine(ROOT)
    dll = worldline._load()
    sample_array = (ctypes.c_float * len(samples))(*samples)
    f0_ptr = ctypes.POINTER(ctypes.c_double)()
    length = dll.F0(sample_array, len(samples), sample_rate, 10.0, 0, ctypes.byref(f0_ptr))
    return [float(f0_ptr[index]) for index in range(length)] if length > 0 else []


def _midi_to_hz(midi: int) -> float:
    """Convert MIDI to Hz."""

    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


if __name__ == "__main__":
    main()
