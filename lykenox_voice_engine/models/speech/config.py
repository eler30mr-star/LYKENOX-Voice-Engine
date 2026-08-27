"""Configuration for the first LYKENOX-owned neural speech model."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LykenoxSpeechConfig:
    """Compact CPU-oriented acoustic model configuration.

    This is a LYKENOX product contract, not a Piper/Coqui configuration.
    The initial model predicts mel spectrogram frames from Spanish token IDs.
    A separate LYKENOX vocoder stage converts mel frames to waveform audio.
    """

    vocab_size: int = 128
    hidden_size: int = 192
    encoder_layers: int = 4
    encoder_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    mel_bins: int = 80
    max_duration_frames: int = 80
    sample_rate: int = 24000
    hop_length: int = 256
    n_fft: int = 1024

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
