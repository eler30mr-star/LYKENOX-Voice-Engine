"""Target-relative spectral balance objective for the LYKENOX v4.1 vocoder.

The first successful source-filter probe removed the frame-rate carrier but listening
showed too much authority in the fundamental/subgrave region.  This training-only module
compares normalized waveform energy in broad speech bands instead of hard-coding a notch
or an absolute spectral target.

Because the loss is target-relative, a genuinely low-pitched LYKENOX utterance remains
valid; the model is only penalized when its generated band distribution diverges from the
paired real waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


VOCODER_SOURCE_BALANCE_VERSION = "vocoder-source-balance-v1"
SPECTRAL_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 80.0),
    (80.0, 300.0),
    (300.0, 3000.0),
    (3000.0, 12000.0),
)
SPECTRAL_BAND_WEIGHTS: tuple[float, ...] = (0.75, 1.0, 1.5, 1.25)


@dataclass(frozen=True)
class SpectralBandBalanceResult:
    loss: torch.Tensor
    generated_fractions: torch.Tensor
    target_fractions: torch.Tensor


def spectral_band_fractions(
    waveform: torch.Tensor,
    *,
    sample_rate: int = 24000,
) -> torch.Tensor:
    """Return normalized energy fractions in four broad speech bands.

    ``waveform`` must be ``[batch, samples]``.  The final band is clipped to Nyquist so
    the contract remains valid if sample rate changes in a future version.
    """

    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [batch, samples]")
    if waveform.shape[1] < 16:
        raise ValueError("waveform is too short for spectral balance analysis")
    if sample_rate < 1000:
        raise ValueError("sample_rate is implausibly low")

    window = torch.hann_window(
        int(waveform.shape[1]),
        device=waveform.device,
        dtype=waveform.dtype,
    )
    spectrum = torch.fft.rfft(waveform * window.unsqueeze(0), dim=1)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(
        int(waveform.shape[1]),
        d=1.0 / float(sample_rate),
        device=waveform.device,
        dtype=waveform.dtype,
    )
    nyquist = sample_rate / 2.0

    energies: list[torch.Tensor] = []
    for low_hz, high_hz in SPECTRAL_BANDS_HZ:
        high = min(float(high_hz), nyquist)
        if high <= low_hz:
            energies.append(torch.zeros(waveform.shape[0], device=waveform.device, dtype=waveform.dtype))
            continue
        if high >= nyquist:
            mask = (frequencies >= low_hz) & (frequencies <= high)
        else:
            mask = (frequencies >= low_hz) & (frequencies < high)
        energies.append(power[:, mask].sum(dim=1))

    stacked = torch.stack(energies, dim=1).clamp_min(1e-12)
    total = stacked.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return stacked / total


def target_relative_spectral_balance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int = 24000,
) -> SpectralBandBalanceResult:
    """Compare generated and target *relative* energy distributions.

    Log fractions give useful gradient to quiet upper bands without letting raw signal
    amplitude dominate.  The 300-3000 Hz speech/formant region receives the strongest
    weight because that was the region missing from the first v4 listening examples.
    """

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [batch, samples]")

    generated = spectral_band_fractions(prediction, sample_rate=sample_rate)
    with torch.no_grad():
        truth = spectral_band_fractions(target, sample_rate=sample_rate)

    weights = torch.tensor(
        SPECTRAL_BAND_WEIGHTS,
        device=prediction.device,
        dtype=prediction.dtype,
    ).view(1, -1)
    difference = F.smooth_l1_loss(
        torch.log(generated.clamp_min(1e-6)),
        torch.log(truth.clamp_min(1e-6)),
        reduction="none",
    )
    loss = (difference * weights).sum(dim=1) / weights.sum()
    return SpectralBandBalanceResult(
        loss=loss.mean(),
        generated_fractions=generated,
        target_fractions=truth,
    )
