"""Configuration for the first LYKENOX-owned neural speech model."""

from __future__ import annotations

from dataclasses import dataclass, asdict


FRAME_CONTEXT_NONE = "none"
FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1 = "token-progress-conv-v1"


@dataclass(frozen=True)
class LykenoxSpeechConfig:
    """Compact CPU-oriented acoustic model configuration.

    This is a LYKENOX product contract, not a Piper/Coqui configuration.
    The acoustic model predicts mel, F0, voicing and duration from Spanish token IDs.
    A separate LYKENOX vocoder stage converts frame acoustics to waveform audio.

    ``frame_context_version`` is deliberately backward-compatible. Historical v0/v1
    checkpoints that predate this field load with ``none`` and therefore reconstruct
    their original piecewise-constant post-regulation architecture exactly. New acoustic
    training may opt into ``token-progress-conv-v1`` without making those checkpoints
    silently incompatible.

    ``feature_cache_dict`` intentionally preserves the exact pre-frame-context config
    payload used by existing mel-v1 and pitch-v1 cache identities. Acoustic architecture
    choices must not invalidate deterministic waveform-derived feature caches.
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
    frame_context_version: str = FRAME_CONTEXT_NONE
    frame_context_layers: int = 3
    frame_context_kernel_size: int = 5

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    def feature_cache_dict(self) -> dict[str, int | float]:
        """Return the historical config payload that owns mel/pitch cache identity.

        The first mel-v1 cache used the full then-current speech config, before frame
        context fields existed. Reconstruct that exact payload so the accepted caches are
        reused across acoustic architecture revisions instead of being duplicated.
        """

        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "encoder_layers": self.encoder_layers,
            "encoder_heads": self.encoder_heads,
            "ff_multiplier": self.ff_multiplier,
            "dropout": self.dropout,
            "mel_bins": self.mel_bins,
            "max_duration_frames": self.max_duration_frames,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
        }
