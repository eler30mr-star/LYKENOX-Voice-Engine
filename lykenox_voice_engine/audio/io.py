"""Product-side audio I/O helpers for LYKENOX Voice Engine.

Use libsndfile via soundfile for local PCM WAV decoding so training does not depend
on torchaudio's optional TorchCodec loader. Tensor operations may still use Torch
and torchaudio after bytes have been decoded locally.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio as float32 tensor shaped [channels, samples].

    This function is the stable LYKENOX audio-loading boundary for local WAV data.
    It intentionally avoids ``torchaudio.load`` because recent torchaudio builds may
    require the optional TorchCodec package even for WAV decoding.
    """

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    waveform = torch.from_numpy(data.T.copy()).contiguous()
    return waveform, int(sample_rate)
