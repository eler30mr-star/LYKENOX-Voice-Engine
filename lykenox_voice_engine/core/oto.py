"""UTAU-style oto.ini parsing and lightweight timing estimation."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from lykenox_voice_engine.core.pcm import rms


@dataclass(frozen=True)
class OtoEntry:
    """One UTAU-style timing row."""

    wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float

    def to_line(self) -> str:
        """Serialize the entry to oto.ini syntax."""

        values = [self.offset, self.consonant, self.cutoff, self.preutterance, self.overlap]
        encoded = ",".join(_format_number(value) for value in values)
        return f"{self.wav}={self.alias},{encoded}"


def parse_oto(path: Path) -> dict[str, OtoEntry]:
    """Parse an oto.ini file into entries keyed by alias."""

    if not path.exists():
        return {}
    entries: dict[str, OtoEntry] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        wav_name, payload = stripped.split("=", 1)
        parts = [part.strip() for part in payload.split(",")]
        if len(parts) != 6:
            raise ValueError(f"oto.ini invalido: {line}")
        alias = parts[0].lower()
        entries[alias] = OtoEntry(
            wav=wav_name.strip(),
            alias=alias,
            offset=float(parts[1]),
            consonant=float(parts[2]),
            cutoff=float(parts[3]),
            preutterance=float(parts[4]),
            overlap=float(parts[5]),
        )
    return entries


def write_oto(path: Path, entries: list[OtoEntry]) -> None:
    """Write entries to oto.ini with deterministic ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entry.to_line() for entry in entries) + "\n", encoding="utf-8")


def estimate_oto_entry(wav_path: Path, alias: str) -> OtoEntry:
    """Estimate initial OTO markers from a WAV energy threshold."""

    with wave.open(str(wav_path), "rb") as reader:
        sample_rate = reader.getframerate()
        width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    window = max(1, int(sample_rate * 0.01))
    threshold = max(400, rms(frames, width) * 0.12)
    start_frame = 0
    for frame_index in range(0, len(frames), window * width):
        chunk = frames[frame_index : frame_index + window * width]
        if chunk and rms(chunk, width) >= threshold:
            start_frame = frame_index // width
            break
    offset = round(start_frame / sample_rate * 1000.0, 3)
    preutterance = round(offset + 60.0, 3)
    overlap = round(offset + 25.0, 3)
    consonant = round(max(80.0, preutterance - offset), 3)
    return OtoEntry(wav=wav_path.name, alias=alias.lower(), offset=offset, consonant=consonant, cutoff=-50.0, preutterance=preutterance, overlap=overlap)


def _format_number(value: float) -> str:
    """Format millisecond values compactly for oto.ini."""

    return f"{value:.3f}".rstrip("0").rstrip(".")
