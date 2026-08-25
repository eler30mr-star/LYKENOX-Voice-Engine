"""Generate WORLDLINE-compatible oto.ini timings from existing WAV samples."""

from __future__ import annotations

import json
import math
import sys
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.core.oto import OtoEntry, parse_oto, write_oto

FRAME_MS = 10.0
HOP_MS = 5.0
PURE_VOWELS = {"a", "e", "i", "o", "u", "- a", "- e", "- i", "- o", "- u"}


@dataclass(frozen=True)
class WavAnalysis:
    """Timing analysis for one voicebank WAV."""

    alias: str
    wav: str
    duration_ms: float
    sample_rate: int
    rms: float
    onset_ms: float
    stable_ms: float
    useful_end_ms: float
    old_useful_ms: float
    new_useful_ms: float
    entry: dict[str, float | str]


def main() -> None:
    """Analyze voicebank WAVs and write oto_worldline.ini."""

    voicebank_dir = ROOT / "profiles" / "lykenox" / "voicebank"
    wav_dir = voicebank_dir / "wav"
    old_oto = parse_oto(voicebank_dir / "oto.ini")
    analyses: list[WavAnalysis] = []
    entries: list[OtoEntry] = []
    for wav_path in sorted(path for path in wav_dir.glob("*.wav") if path.stat().st_size > 44):
        alias = wav_path.stem.lower()
        old = old_oto.get(alias)
        entry, analysis = analyze_wav(wav_path, alias, old)
        entries.append(entry)
        analyses.append(analysis)

    output = voicebank_dir / "oto_worldline.ini"
    write_oto(output, entries)
    report_path = voicebank_dir / "oto_worldline_report.json"
    report_path.write_text(
        json.dumps([asdict(item) for item in analyses], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    examples = {
        alias: next(item for item in entries if item.alias == alias).to_line()
        for alias in ["a", "ba", "bai", "con"]
        if any(item.alias == alias for item in entries)
    }
    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(report_path),
                "count": len(entries),
                "examples": examples,
                "a_analysis": asdict(next(item for item in analyses if item.alias == "a")),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def analyze_wav(wav_path: Path, alias: str, old: OtoEntry | None) -> tuple[OtoEntry, WavAnalysis]:
    """Analyze one WAV and return a WORLDLINE-compatible OTO entry."""

    sample_rate, samples = read_wav(wav_path)
    duration_ms = len(samples) / sample_rate * 1000.0
    frame = max(1, int(sample_rate * FRAME_MS / 1000.0))
    hop = max(1, int(sample_rate * HOP_MS / 1000.0))
    rms_values = []
    for start in range(0, max(1, len(samples) - frame), hop):
        chunk = samples[start : start + frame]
        rms_values.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
    peak_rms = max(rms_values, default=0.0)
    noise = sorted(rms_values[: min(20, len(rms_values))] or [0.0])[len(rms_values[: min(20, len(rms_values))]) // 2]
    threshold = max(noise * 3.0, peak_rms * 0.08, 0.003)
    onset_frame = first_sustained(rms_values, threshold)
    end_frame = last_sustained(rms_values, threshold)
    onset_ms = onset_frame * HOP_MS
    useful_end_ms = min(duration_ms, (end_frame * HOP_MS) + FRAME_MS)
    stable_ms = find_stable_start(rms_values, onset_frame, threshold) * HOP_MS

    if alias in PURE_VOWELS:
        consonant = min(40.0, max(20.0, stable_ms - onset_ms))
        preutterance = max(onset_ms, min(stable_ms, onset_ms + consonant * 0.75))
        overlap = min(20.0, max(5.0, preutterance * 0.35))
    else:
        consonant = min(180.0, max(60.0, stable_ms - onset_ms))
        preutterance = max(onset_ms, min(stable_ms, onset_ms + consonant * 0.75))
        overlap = min(50.0, max(15.0, preutterance * 0.35))

    offset = max(0.0, onset_ms)
    min_useful = 500.0 if duration_ms >= 1000 else max(100.0, duration_ms * 0.55)
    useful_end_ms = max(useful_end_ms, min(duration_ms, offset + min_useful))
    cutoff = max(0.0, duration_ms - useful_end_ms)

    entry = OtoEntry(
        wav=wav_path.name,
        alias=alias,
        offset=round(offset, 3),
        consonant=round(consonant, 3),
        cutoff=round(cutoff, 3),
        preutterance=round(preutterance, 3),
        overlap=round(overlap, 3),
    )
    old_useful = -old.cutoff if old and old.cutoff < 0 else (
        duration_ms - old.offset - old.cutoff if old else 0.0
    )
    new_useful = duration_ms - entry.offset - entry.cutoff
    analysis = WavAnalysis(
        alias=alias,
        wav=wav_path.name,
        duration_ms=round(duration_ms, 3),
        sample_rate=sample_rate,
        rms=round(math.sqrt(sum(value * value for value in samples) / len(samples)), 6),
        onset_ms=round(onset_ms, 3),
        stable_ms=round(stable_ms, 3),
        useful_end_ms=round(useful_end_ms, 3),
        old_useful_ms=round(old_useful, 3),
        new_useful_ms=round(new_useful, 3),
        entry=entry.__dict__,
    )
    return entry, analysis


def first_sustained(values: list[float], threshold: float) -> int:
    """Return first index with sustained energy above threshold."""

    for index in range(len(values)):
        if sum(1 for value in values[index : index + 3] if value >= threshold) >= 2:
            return index
    return 0


def last_sustained(values: list[float], threshold: float) -> int:
    """Return last index with sustained energy above threshold."""

    for index in range(len(values) - 1, -1, -1):
        if sum(1 for value in values[max(0, index - 2) : index + 1] if value >= threshold) >= 2:
            return index
    return max(0, len(values) - 1)


def find_stable_start(values: list[float], onset: int, threshold: float) -> int:
    """Find the first likely stable frame after attack."""

    target = max(threshold * 1.5, max(values, default=0.0) * 0.18)
    for index in range(onset, len(values)):
        window = values[index : index + 5]
        if len(window) == 5 and sum(1 for value in window if value >= target) >= 4:
            return index
    return onset


def read_wav(path: Path) -> tuple[int, list[float]]:
    """Read PCM16 WAV as mono normalized floats."""

    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        raw = reader.readframes(reader.getnframes())
    if width != 2:
        raise ValueError(f"Expected PCM16 WAV: {path}")
    values = array("h")
    values.frombytes(raw)
    if channels == 1:
        return sample_rate, [value / 32768.0 for value in values]
    mono = []
    for index in range(0, len(values), channels):
        mono.append(sum(values[index : index + channels]) / channels / 32768.0)
    return sample_rate, mono


if __name__ == "__main__":
    main()
