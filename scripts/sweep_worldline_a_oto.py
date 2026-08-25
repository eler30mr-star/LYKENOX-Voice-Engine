"""Sweep a.wav OTO preutterance values for WORLDLINE-R diagnostics."""

from __future__ import annotations

import ctypes
import sys
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.oto import OtoEntry, parse_oto
from lykenox_voice_engine.engines.worldline_engine import OpenUtauWorldlineEngine
from lykenox_voice_engine.models.notes import NoteEvent


def main() -> None:
    """Render temporary sweep files and print mean F0 by preutterance."""

    wav_dir = ROOT / "profiles" / "lykenox" / "voicebank" / "wav"
    base = parse_oto(ROOT / "profiles" / "lykenox" / "voicebank" / "oto_worldline.ini")
    worldline = OpenUtauWorldlineEngine(ROOT)
    rows = []
    for preutterance in [30, 60, 120, 250, 500, 800, 1200, 1570]:
        oto = dict(base)
        oto["a"] = OtoEntry(
            "a.wav",
            "a",
            1540.0,
            40.0,
            795.0,
            float(preutterance),
            min(20.0, max(5.0, preutterance * 0.35)),
        )
        measured = []
        for midi in [60, 62, 64]:
            path = ROOT / "outputs" / "worldline_pitch_tests" / f"sweep_pre{preutterance}_m{midi}.wav"
            worldline.render_to_path(
                wav_dir,
                oto,
                lambda lyric: [lyric],
                [NoteEvent("a", midi, 0.0, 1.5)],
                120,
                path,
            )
            start_sec = (preutterance + 40.0 + 250.0) / 1000.0
            measured.append(round(mean_f0(path, start_sec), 2))
        rows.append({"preutterance": preutterance, "measured": measured})
    for row in rows:
        print(row)


def mean_f0(path: Path, start_sec: float, duration_sec: float = 0.6) -> float:
    """Measure mean F0 in a stable window using WORLDLINE-R F0 export."""

    sample_rate, samples = read_wav(path)
    worldline = OpenUtauWorldlineEngine(ROOT)
    dll = worldline._load()
    sample_array = (ctypes.c_float * len(samples))(*samples)
    f0_ptr = ctypes.POINTER(ctypes.c_double)()
    length = dll.F0(sample_array, len(samples), sample_rate, 10.0, 0, ctypes.byref(f0_ptr))
    f0 = [float(f0_ptr[index]) for index in range(length)] if length > 0 else []
    voiced = [value for value in f0[int(start_sec * 100) : int((start_sec + duration_sec) * 100)] if value > 0]
    return sum(voiced) / len(voiced) if voiced else 0.0


def read_wav(path: Path) -> tuple[int, list[float]]:
    """Read PCM16 WAV."""

    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        raw = reader.readframes(reader.getnframes())
    values = array("h")
    values.frombytes(raw)
    return sample_rate, [value / 32768.0 for value in values]


if __name__ == "__main__":
    main()
