"""Multipitch voicebank helpers for WORLDLINE/UTAU-style sample selection."""

from __future__ import annotations

import math
import statistics
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from lykenox_voice_engine.core.oto import OtoEntry

LAYERS = ("Low", "Mid", "High")
MICROTEST_ALIASES = ("a", "bai", "la", "con", "mi", "go")


@dataclass(frozen=True)
class PitchCenter:
    """One recording layer center derived from the current voicebank range."""

    layer: str
    midi: int
    note: str
    hz: float
    range_min_midi: int
    range_max_midi: int


@dataclass(frozen=True)
class PitchRangeReport:
    """Measured pitch statistics from the existing monopitch voicebank."""

    measured_count: int
    median_hz: float
    mean_hz: float
    min_hz: float
    max_hz: float
    median_midi: float
    centers: tuple[PitchCenter, ...]


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""

    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(hz: float) -> float:
    """Convert frequency in Hz to fractional MIDI note number."""

    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_note(midi: int) -> str:
    """Return a simple western note name for a MIDI note."""

    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = (midi // 12) - 1
    return f"{names[midi % 12]}{octave}"


def design_pitch_centers(f0_values: list[float]) -> PitchRangeReport:
    """Design Low/Mid/High centers from measured F0 values."""

    valid = sorted(value for value in f0_values if value > 0)
    if not valid:
        raise ValueError("No hay F0 valido para disenar centros multipitch.")
    median_hz = statistics.median(valid)
    median_midi = hz_to_midi(median_hz)
    low_midi = int(round(median_midi))
    centers = (
        _center("Low", low_midi, -99, low_midi + 2),
        _center("Mid", low_midi + 5, low_midi + 3, low_midi + 7),
        _center("High", low_midi + 10, low_midi + 8, 127),
    )
    return PitchRangeReport(
        measured_count=len(valid),
        median_hz=round(median_hz, 2),
        mean_hz=round(sum(valid) / len(valid), 2),
        min_hz=round(min(valid), 2),
        max_hz=round(max(valid), 2),
        median_midi=round(median_midi, 2),
        centers=centers,
    )


def read_prefix_map(path: Path) -> dict[int, str]:
    """Read a numeric UTAU/OpenUtau prefix.map into MIDI-to-suffix rows."""

    if not path.exists():
        return {}
    rows: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw_line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        try:
            midi = int(parts[0])
        except ValueError:
            continue
        rows[midi] = parts[-1].strip()
    return rows


def write_prefix_map(path: Path, centers: tuple[PitchCenter, ...]) -> None:
    """Write suffix rows so OpenUtau/UTAU-style selection can use layers."""

    suffixes = _suffixes_by_midi(centers)
    lines = [
        "# LYKENOX multipitch suffix map.",
        "# Format: midi prefix suffix",
        "# Empty prefix, suffix selects alias_Low / alias_Mid / alias_High.",
    ]
    lines.extend(f"{midi}\t\t{suffix}" for midi, suffix in enumerate(suffixes))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def suffix_for_midi(prefix_map: dict[int, str], midi: int) -> str:
    """Return the configured multipitch suffix for a MIDI note."""

    if midi in prefix_map:
        return prefix_map[midi]
    if not prefix_map:
        return ""
    nearest = min(prefix_map, key=lambda key: abs(key - midi))
    return prefix_map[nearest]


def layer_alias(base_alias: str, suffix: str) -> str:
    """Append a layer suffix to a base alias if one is configured."""

    return f"{base_alias}{suffix}".lower() if suffix else base_alias.lower()


def layer_wav_name(alias: str, layer: str) -> str:
    """Return the WAV filename for one alias in one multipitch layer."""

    return f"{alias}_{layer}.wav"


def flatten_layer_oto(base: dict[str, OtoEntry], layer: str) -> list[OtoEntry]:
    """Convert layer-local OTO entries into alias_Layer entries."""

    suffix = f"_{layer}"
    entries: list[OtoEntry] = []
    for alias, entry in sorted(base.items()):
        entries.append(
            OtoEntry(
                wav=layer_wav_name(alias, layer),
                alias=f"{alias}{suffix}".lower(),
                offset=entry.offset,
                consonant=entry.consonant,
                cutoff=entry.cutoff,
                preutterance=entry.preutterance,
                overlap=entry.overlap,
            )
        )
    return entries


def report_to_dict(report: PitchRangeReport) -> dict[str, object]:
    """Serialize a pitch report for JSON output."""

    return {
        "measured_count": report.measured_count,
        "median_hz": report.median_hz,
        "mean_hz": report.mean_hz,
        "min_hz": report.min_hz,
        "max_hz": report.max_hz,
        "median_midi": report.median_midi,
        "centers": [asdict(center) for center in report.centers],
    }


def wav_duration_sec(path: Path) -> float:
    """Return WAV duration in seconds."""

    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / float(reader.getframerate())


def _center(layer: str, midi: int, range_min: int, range_max: int) -> PitchCenter:
    return PitchCenter(layer, midi, midi_to_note(midi), round(midi_to_hz(midi), 2), range_min, range_max)


def _suffixes_by_midi(centers: tuple[PitchCenter, ...]) -> list[str]:
    suffixes: list[str] = []
    for midi in range(128):
        layer = min(centers, key=lambda center: abs(center.midi - midi)).layer
        suffixes.append(f"_{layer}")
    return suffixes
