"""Configuration contract for the first LYKENOX-owned neural vocoder prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod


@dataclass(frozen=True)
class LykenoxVocoderConfig:
    """Compact non-autoregressive mel-to-waveform generator for CPU feasibility work.

    This is a LYKENOX model contract, not a wrapper around an external vocoder.
    The initial factors multiply to the speech hop length so one mel frame expands
    to exactly 256 waveform samples at 24 kHz.
    """

    mel_bins: int = 80
    sample_rate: int = 24000
    hop_length: int = 256
    channels: int = 128
    upsample_factors: tuple[int, ...] = (8, 8, 4)
    residual_kernel_size: int = 3
    residual_dilations: tuple[int, ...] = (1, 3)

    def __post_init__(self) -> None:
        if self.mel_bins < 1 or self.sample_rate < 1 or self.hop_length < 1:
            raise ValueError("mel/audio dimensions must be positive")
        if self.channels < 16:
            raise ValueError("channels is unrealistically small")
        if not self.upsample_factors or any(factor < 2 for factor in self.upsample_factors):
            raise ValueError("upsample_factors must contain factors >= 2")
        if prod(self.upsample_factors) != self.hop_length:
            raise ValueError(
                "LYKENOX vocoder upsample factors must multiply exactly to hop_length"
            )
        if self.residual_kernel_size < 3 or self.residual_kernel_size % 2 == 0:
            raise ValueError("residual_kernel_size must be odd and >= 3")
        if not self.residual_dilations or any(value < 1 for value in self.residual_dilations):
            raise ValueError("residual_dilations must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
