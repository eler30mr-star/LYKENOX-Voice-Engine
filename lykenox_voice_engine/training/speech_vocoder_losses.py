"""LYKENOX-owned vocoder training losses.

The persistent recipe uses a stable spectral reconstruction objective first, then adds
lightweight adversarial and feature-matching terms. These functions are training-only;
the final runtime needs neither discriminator nor loss code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lykenox_voice_engine.models.vocoder import DiscriminatorOutput


VOCODER_LOSS_RECIPE_VERSION = "vocoder-loss-v1"
STFT_RESOLUTIONS: tuple[tuple[int, int, int], ...] = (
    (256, 64, 256),
    (512, 128, 512),
    (1024, 256, 1024),
)


@dataclass(frozen=True)
class VocoderReconstructionLoss:
    total: torch.Tensor
    waveform_l1: torch.Tensor
    spectral_convergence: torch.Tensor
    log_magnitude: torch.Tensor


def multi_resolution_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    waveform_weight: float = 0.10,
) -> VocoderReconstructionLoss:
    """Waveform + multi-resolution STFT reconstruction loss.

    The spectral convergence term tracks relative magnitude error and log-magnitude L1
    gives strong pressure on quiet harmonics/formants. The waveform term is deliberately
    small because exact sample phase is not the main perceptual target.
    """

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share shape [batch, samples]")
    waveform_l1 = F.l1_loss(prediction, target)
    convergence_terms: list[torch.Tensor] = []
    logmag_terms: list[torch.Tensor] = []
    for n_fft, hop_length, win_length in STFT_RESOLUTIONS:
        window = torch.hann_window(
            win_length,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        pred = torch.stft(
            prediction,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs().clamp_min(1e-5)
        truth = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        ).abs().clamp_min(1e-5)
        difference = torch.linalg.vector_norm(truth - pred)
        reference = torch.linalg.vector_norm(truth).clamp_min(1e-6)
        convergence_terms.append(difference / reference)
        logmag_terms.append(F.l1_loss(torch.log(pred), torch.log(truth)))

    spectral_convergence = torch.stack(convergence_terms).mean()
    log_magnitude = torch.stack(logmag_terms).mean()
    total = (
        waveform_weight * waveform_l1
        + spectral_convergence
        + log_magnitude
    )
    return VocoderReconstructionLoss(
        total=total,
        waveform_l1=waveform_l1,
        spectral_convergence=spectral_convergence,
        log_magnitude=log_magnitude,
    )


def discriminator_hinge_loss(
    real: DiscriminatorOutput,
    fake: DiscriminatorOutput,
) -> torch.Tensor:
    if len(real.scores) != len(fake.scores):
        raise ValueError("real/fake discriminator scale count mismatch")
    terms = [
        F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
        for real_score, fake_score in zip(real.scores, fake.scores, strict=True)
    ]
    return torch.stack(terms).mean()


def generator_adversarial_loss(fake: DiscriminatorOutput) -> torch.Tensor:
    return torch.stack([-score.mean() for score in fake.scores]).mean()


def feature_matching_loss(
    real: DiscriminatorOutput,
    fake: DiscriminatorOutput,
) -> torch.Tensor:
    if len(real.feature_maps) != len(fake.feature_maps):
        raise ValueError("real/fake discriminator feature scale mismatch")
    terms: list[torch.Tensor] = []
    for real_scale, fake_scale in zip(real.feature_maps, fake.feature_maps, strict=True):
        if len(real_scale) != len(fake_scale):
            raise ValueError("real/fake discriminator feature depth mismatch")
        for real_feature, fake_feature in zip(real_scale, fake_scale, strict=True):
            terms.append(F.l1_loss(fake_feature, real_feature.detach()))
    if not terms:
        raise RuntimeError("No discriminator features available for feature matching")
    return torch.stack(terms).mean()
