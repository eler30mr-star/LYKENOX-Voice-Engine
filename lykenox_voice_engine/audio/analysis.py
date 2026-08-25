"""Lightweight audio inspection helpers for dataset imports."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".opus"}


@dataclass(frozen=True)
class AudioInfo:
    """Basic dataset audio metadata shown in the desktop UI."""

    path: Path
    duration: float | None
    sample_rate: int | None
    channels: int | None
    peak: float | None
    status: str


def inspect_audio(path: Path) -> AudioInfo:
    """Inspect a dataset audio file without modifying the original."""

    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return AudioInfo(path, None, None, None, None, "unsupported")
    if path.suffix.lower() != ".wav":
        return AudioInfo(path, None, None, None, None, "needs_prepare")
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            duration = frames / rate if rate else 0.0
        return AudioInfo(path, duration, rate, channels, None, "ok")
    except wave.Error:
        return AudioInfo(path, None, None, None, None, "invalid")
