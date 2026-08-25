"""Small PCM metrics used by voicebank validation and OTO estimation."""

from __future__ import annotations

import struct


def rms(frames: bytes, sample_width: int) -> int:
    """Return the integer RMS amplitude for signed little-endian PCM."""

    samples = _samples(frames, sample_width)
    if not samples:
        return 0
    return int((sum(sample * sample for sample in samples) / len(samples)) ** 0.5)


def peak(frames: bytes, sample_width: int) -> int:
    """Return the largest absolute PCM sample amplitude."""

    samples = _samples(frames, sample_width)
    return max((abs(sample) for sample in samples), default=0)


def _samples(frames: bytes, sample_width: int) -> list[int]:
    """Decode mono PCM samples for the widths accepted by wave."""

    if sample_width == 1:
        return [sample - 128 for sample in frames]
    if sample_width == 2:
        count = len(frames) // 2
        return list(struct.unpack(f"<{count}h", frames[: count * 2])) if count else []
    if sample_width == 3:
        values: list[int] = []
        for offset in range(0, len(frames) - 2, 3):
            value = int.from_bytes(frames[offset : offset + 3], "little", signed=False)
            values.append(value - (1 << 24) if value & (1 << 23) else value)
        return values
    if sample_width == 4:
        count = len(frames) // 4
        return list(struct.unpack(f"<{count}i", frames[: count * 4])) if count else []
    raise ValueError(f"sample width no soportado: {sample_width}")
