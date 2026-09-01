"""Structural smoke for the owned frame-rate cepstral predictor.

This gate is intentionally before optimizer authorization.  It may instantiate the predictor
and use autograd, but it creates no optimizer, performs no parameter update, saves no
checkpoint, and makes no product-quality claim.

The smoke proves that neutral initialization preserves the fixed renderer's exact identity
point, output shape remains frame-rate only, gradients are finite/connected, and integrating
the neutral predictor with the proven renderer cannot introduce frame-grid excess.
"""

from __future__ import annotations

import json
import math

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_v1 import (
    CEPSTRAL_ORDER,
    MEL_BINS,
    PREDICTOR_ARCHITECTURE,
    LykenoxFrameRateCepstralPredictorV1,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    frame_grid_artifact_excess_metrics,
)


SMOKE_VERSION = "owned-frame-rate-cepstral-predictor-smoke-v1"
ARCHITECTURE_CONTRACT_VERSION = "owned-vocoder-architecture-contract-v1"
RENDERER_VERSION = "owned-minimum-phase-time-varying-renderer-v1"
OPTIMIZER_CREATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False


def _inputs(*, batch: int = 2, frames: int = 48) -> tuple[torch.Tensor, ...]:
    frame = torch.arange(frames, dtype=torch.float32)
    mel_bins = torch.arange(MEL_BINS, dtype=torch.float32)
    mel = (
        -4.0
        + 0.35 * torch.sin(frame[:, None] * 0.17 + mel_bins[None, :] * 0.07)
        + 0.12 * torch.cos(frame[:, None] * 0.41 - mel_bins[None, :] * 0.03)
    )
    mel = mel.unsqueeze(0).repeat(batch, 1, 1)
    if batch > 1:
        mel[1] = mel[1] + 0.05 * torch.sin(frame[:, None] * 0.29)

    f0 = (175.0 + 24.0 * torch.sin(frame * 0.19)).unsqueeze(0).repeat(batch, 1)
    voiced = torch.ones(batch, frames, dtype=torch.float32)
    periodicity = (0.82 + 0.08 * torch.sin(frame * 0.13)).clamp(0.0, 1.0)
    periodicity = periodicity.unsqueeze(0).repeat(batch, 1)
    if batch > 1:
        voiced[1, ::7] = 0.0
        f0[1, ::7] = 0.0
        periodicity[1, ::7] = 0.0
    return mel, f0, voiced, periodicity


def _neutral_case() -> dict[str, object]:
    torch.manual_seed(20260901)
    model = LykenoxFrameRateCepstralPredictorV1()
    mel, f0, voiced, periodicity = _inputs()
    cepstrum = model(mel, f0, voiced, periodicity)
    waveform, excitation = renderer.render_owned_minimum_phase_vocoder_path(
        cepstrum,
        f0,
        voiced,
        periodicity,
        noise_seed=73,
    )
    identity_error = float((waveform - excitation).abs().max())
    grid = frame_grid_artifact_excess_metrics(
        waveform,
        excitation,
        sample_rate=renderer.SAMPLE_RATE,
        hop_length=renderer.HOP_LENGTH,
    )
    return {
        "predictor_output_shape": list(cepstrum.shape),
        "expected_predictor_output_shape": [mel.shape[0], mel.shape[1], CEPSTRAL_ORDER],
        "maximum_abs_initial_cepstrum": float(cepstrum.abs().max()),
        "neutral_initialization_exact": bool(torch.count_nonzero(cepstrum) == 0),
        "renderer_identity_max_abs_error": identity_error,
        "renderer_identity_exact": identity_error == 0.0,
        "expected_waveform_samples": int(mel.shape[1] * renderer.HOP_LENGTH),
        "actual_waveform_samples": int(waveform.shape[-1]),
        "exact_output_length": int(waveform.shape[-1]) == int(mel.shape[1] * renderer.HOP_LENGTH),
        "hop_autocorrelation_excess": float(grid.hop_autocorrelation_excess.max()),
        "double_hop_autocorrelation_excess": float(grid.double_hop_autocorrelation_excess.max()),
        "grid_harmonic_power_fraction_excess": float(
            grid.grid_harmonic_power_fraction_excess.max()
        ),
        "severe_grid_excess": bool(grid.severe_grid_excess.any()),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _gradient_case() -> dict[str, object]:
    torch.manual_seed(20260902)
    model = LykenoxFrameRateCepstralPredictorV1()
    mel, f0, voiced, periodicity = _inputs(batch=1, frames=24)

    # At neutral init only the zero-initialized output head needs immediate non-zero gradient.
    prediction = model(mel, f0, voiced, periodicity)
    frame = torch.arange(prediction.shape[1], dtype=prediction.dtype)
    target = torch.zeros_like(prediction)
    target[..., 0] = 0.08 * torch.sin(frame * 0.31)
    target[..., 1] = 0.05 * torch.cos(frame * 0.17)
    target[..., 3] = -0.03 * torch.sin(frame * 0.43)
    loss = (prediction - target).square().mean()
    neutral_gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    neutral_finite = all(
        gradient is None or bool(torch.isfinite(gradient).all())
        for gradient in neutral_gradients
    )
    head_weight_gradient = neutral_gradients[-2]
    head_bias_gradient = neutral_gradients[-1]
    head_gradient_nonzero = (
        head_weight_gradient is not None
        and head_bias_gradient is not None
        and float(head_weight_gradient.norm()) > 0.0
        and float(head_bias_gradient.norm()) > 0.0
    )

    # Diagnostic-only deterministic head perturbation proves the upstream frame network is
    # connected once the head leaves its exact identity initialization.  No optimizer or
    # training step is created or executed.
    with torch.no_grad():
        index = torch.arange(model.cepstral_projection.weight.numel(), dtype=torch.float32)
        pattern = 1e-4 * torch.sin(index * math.sqrt(2.0))
        model.cepstral_projection.weight.copy_(pattern.view_as(model.cepstral_projection.weight))
    connected_prediction = model(mel, f0, voiced, periodicity)
    connected_loss = (connected_prediction - target).square().mean()
    connected_gradients = torch.autograd.grad(
        connected_loss,
        tuple(model.parameters()),
        allow_unused=True,
    )
    connected_all_present = all(gradient is not None for gradient in connected_gradients)
    connected_all_finite = all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in connected_gradients
    )
    connected_nonzero_tensors = sum(
        1
        for gradient in connected_gradients
        if gradient is not None and float(gradient.norm()) > 0.0
    )
    return {
        "neutral_loss": float(loss),
        "neutral_gradients_finite": neutral_finite,
        "neutral_output_head_gradient_nonzero": bool(head_gradient_nonzero),
        "connected_probe_loss": float(connected_loss),
        "connected_gradients_all_present": connected_all_present,
        "connected_gradients_all_finite": connected_all_finite,
        "connected_nonzero_gradient_tensor_count": int(connected_nonzero_tensors),
        "trainable_parameter_tensor_count": int(len(connected_gradients)),
        "deterministic_head_perturbation_only": True,
        "parameter_update_executed": False,
    }


def run_smoke() -> dict[str, object]:
    neutral = _neutral_case()
    gradients = _gradient_case()
    gates = {
        "frame_rate_shape_exact": neutral["predictor_output_shape"]
        == neutral["expected_predictor_output_shape"],
        "neutral_initialization_exact": bool(neutral["neutral_initialization_exact"]),
        "renderer_identity_exact": bool(neutral["renderer_identity_exact"]),
        "exact_output_length": bool(neutral["exact_output_length"]),
        "no_severe_grid_excess_at_neutral_init": not bool(neutral["severe_grid_excess"]),
        "neutral_gradients_finite": bool(gradients["neutral_gradients_finite"]),
        "neutral_output_head_gradient_nonzero": bool(
            gradients["neutral_output_head_gradient_nonzero"]
        ),
        "connected_gradients_all_present": bool(gradients["connected_gradients_all_present"]),
        "connected_gradients_all_finite": bool(gradients["connected_gradients_all_finite"]),
        "all_parameter_tensors_receive_nonzero_connected_probe_gradient": int(
            gradients["connected_nonzero_gradient_tensor_count"]
        )
        == int(gradients["trainable_parameter_tensor_count"]),
    }
    status = "pass" if all(gates.values()) else "fail"
    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "renderer_version": RENDERER_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "neutral": neutral,
        "gradients": gradients,
        "gates": gates,
        "model_instantiated": True,
        "optimizer_created": False,
        "training_started": False,
        "parameter_update_executed": False,
        "checkpoint_loaded": False,
        "checkpoint_saved": False,
        "persistent_training_authorized": False,
        "new_vocoder_checkpoint_authorized": False,
        "next_gate": (
            "review_owned_frame_rate_cepstral_predictor_before_authorizing_bounded_optimizer_smoke"
            if status == "pass"
            else "revise_owned_frame_rate_cepstral_predictor_before_optimizer"
        ),
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
