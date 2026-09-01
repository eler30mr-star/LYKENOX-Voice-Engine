"""Read-only structural audit for the fixed LYKENOX minimum-phase renderer.

This audit creates no neural model, optimizer, checkpoint, trainer, or product artifact. It
verifies only the fixed renderer selected by ``owned-vocoder-architecture-contract-v1``:
cepstral factorization, exact output length, flat-envelope identity, source suppression when
the filter is attenuating, deterministic excitation, and absence of severe frame-grid excess.

Grid safety uses the existing paired excess detector rather than absolute periodicity because
a naturally voiced excitation can itself be periodic near the 256-sample mel hop.
"""

from __future__ import annotations

import json
import math

import torch

from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    frame_grid_artifact_excess_metrics,
    frame_grid_artifact_metrics,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer


AUDIT_VERSION = "owned-minimum-phase-renderer-safety-audit-v1"
ARCHITECTURE_CONTRACT_VERSION = "owned-vocoder-architecture-contract-v1"
MODEL_INSTANTIATION_AUTHORIZED = False
OPTIMIZER_CREATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False


def _factorization_case() -> dict[str, float | bool]:
    cepstrum = torch.zeros(3, renderer.CEPSTRAL_ORDER, dtype=torch.float64)
    cepstrum[0, 0] = 0.10
    cepstrum[0, 1] = 0.05
    cepstrum[0, 2] = -0.03
    cepstrum[0, 5] = 0.02
    cepstrum[1, 0] = -0.18
    cepstrum[1, 3] = 0.04
    cepstrum[1, 11] = -0.015
    cepstrum[2, 0] = 0.03
    cepstrum[2, 7] = 0.025
    cepstrum[2, 19] = -0.012

    impulse = renderer.one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum)
    represented = renderer.cepstral_log_magnitude(cepstrum)
    reconstructed = torch.log(
        torch.fft.rfft(impulse, n=renderer.N_FFT, dim=-1).abs().clamp_min(1e-30)
    )
    maximum_error = float((represented - reconstructed).abs().max())
    oracle_roundtrip = renderer.reference_log_magnitude_to_one_sided_cepstrum(represented)
    oracle_error = float((oracle_roundtrip - cepstrum).abs().max())
    return {
        "maximum_log_magnitude_factorization_error": maximum_error,
        "maximum_reference_oracle_roundtrip_error": oracle_error,
        "factorization_exact": maximum_error <= 1e-10,
        "reference_oracle_representation_exact": oracle_error <= 1e-10,
    }


def _identity_and_bypass_case() -> dict[str, float | int | bool]:
    frame_count = 10
    sample_count = frame_count * renderer.HOP_LENGTH
    sample_index = torch.arange(sample_count, dtype=torch.float64)
    excitation = (
        torch.sin(2.0 * math.pi * 173.0 * sample_index / renderer.SAMPLE_RATE)
        + 0.17 * torch.cos(2.0 * math.pi * 677.0 * sample_index / renderer.SAMPLE_RATE)
    ).unsqueeze(0)

    flat = torch.zeros(
        1,
        frame_count,
        renderer.CEPSTRAL_ORDER,
        dtype=torch.float64,
    )
    identity = renderer.render_time_varying_minimum_phase(excitation, flat)
    identity_error = float((identity - excitation).abs().max())

    attenuating = flat.clone()
    attenuation_log = -6.0
    attenuating[..., 0] = attenuation_log
    suppressed = renderer.render_time_varying_minimum_phase(excitation, attenuating)
    expected = excitation * math.exp(attenuation_log)
    suppression_error = float((suppressed - expected).abs().max())
    rms_ratio = float(
        suppressed.square().mean().sqrt()
        / excitation.square().mean().sqrt().clamp_min(1e-30)
    )
    return {
        "frame_count": frame_count,
        "expected_sample_count": sample_count,
        "actual_sample_count": int(identity.shape[-1]),
        "flat_envelope_max_abs_identity_error": identity_error,
        "attenuating_filter_expected_scale": math.exp(attenuation_log),
        "attenuating_filter_measured_rms_ratio": rms_ratio,
        "attenuating_filter_max_abs_expected_error": suppression_error,
        "flat_envelope_exact_identity": identity_error <= 1e-12,
        "source_bypass_absent": suppression_error <= 1e-12 and rms_ratio < 0.003,
        "exact_output_length": int(identity.shape[-1]) == sample_count,
    }


def _irregular_cepstrum(frame_count: int) -> torch.Tensor:
    frame = torch.arange(frame_count, dtype=torch.float64)
    cepstrum = torch.zeros(
        1,
        frame_count,
        renderer.CEPSTRAL_ORDER,
        dtype=torch.float64,
    )
    cepstrum[0, :, 0] = 0.05 * torch.sin(frame * math.sqrt(2.0))
    cepstrum[0, :, 1] = 0.06 * torch.sin(frame * 0.731)
    cepstrum[0, :, 2] = 0.04 * torch.cos(frame * math.sqrt(3.0))
    cepstrum[0, :, 3] = 0.03 * torch.sin(frame * 1.137)
    cepstrum[0, :, 5] = 0.02 * torch.cos(frame * 0.419)
    return cepstrum


def _grid_case(*, voiced_case: bool) -> dict[str, float | bool]:
    frame_count = 48
    frame = torch.arange(frame_count, dtype=torch.float64)
    if voiced_case:
        f0 = (180.0 + 25.0 * torch.sin(frame * 0.21) + 12.0 * torch.sin(frame * 0.73)).unsqueeze(0)
        voiced = torch.ones_like(f0)
        periodicity = (0.82 + 0.08 * torch.sin(frame * 0.17)).unsqueeze(0).clamp(0.0, 1.0)
        seed = 29
    else:
        f0 = torch.zeros(1, frame_count, dtype=torch.float64)
        voiced = torch.zeros_like(f0)
        periodicity = torch.zeros_like(f0)
        seed = 11

    cepstrum = _irregular_cepstrum(frame_count)
    candidate, excitation = renderer.render_owned_minimum_phase_vocoder_path(
        cepstrum,
        f0,
        voiced,
        periodicity,
        noise_seed=seed,
    )
    excess = frame_grid_artifact_excess_metrics(
        candidate,
        excitation,
        sample_rate=renderer.SAMPLE_RATE,
        hop_length=renderer.HOP_LENGTH,
    )
    absolute = frame_grid_artifact_metrics(
        candidate,
        sample_rate=renderer.SAMPLE_RATE,
        hop_length=renderer.HOP_LENGTH,
    )
    return {
        "hop_autocorrelation_excess": float(excess.hop_autocorrelation_excess.max()),
        "double_hop_autocorrelation_excess": float(excess.double_hop_autocorrelation_excess.max()),
        "grid_harmonic_power_fraction_excess": float(excess.grid_harmonic_power_fraction_excess.max()),
        "candidate_hop_autocorrelation": float(absolute.hop_autocorrelation.abs().max()),
        "candidate_double_hop_autocorrelation": float(absolute.double_hop_autocorrelation.abs().max()),
        "candidate_grid_harmonic_power_fraction": float(absolute.grid_harmonic_power_fraction.max()),
        "severe_grid_excess": bool(excess.severe_grid_excess.any()),
        "exact_output_length": int(candidate.shape[-1]) == frame_count * renderer.HOP_LENGTH,
    }


def _determinism_case() -> dict[str, float | bool]:
    frame_count = 16
    f0 = torch.zeros(1, frame_count, dtype=torch.float64)
    voiced = torch.zeros_like(f0)
    periodicity = torch.zeros_like(f0)
    first = renderer.build_neutral_excitation(f0, voiced, periodicity, noise_seed=37)
    second = renderer.build_neutral_excitation(f0, voiced, periodicity, noise_seed=37)
    other = renderer.build_neutral_excitation(f0, voiced, periodicity, noise_seed=38)
    same_error = float((first - second).abs().max())
    other_difference = float((first - other).abs().max())
    return {
        "same_seed_max_abs_error": same_error,
        "different_seed_max_abs_difference": other_difference,
        "deterministic_same_seed": same_error == 0.0,
        "seed_changes_aperiodic_realization": other_difference > 1e-3,
    }


def run_audit() -> dict[str, object]:
    factorization = _factorization_case()
    identity = _identity_and_bypass_case()
    unvoiced_grid = _grid_case(voiced_case=False)
    voiced_grid = _grid_case(voiced_case=True)
    determinism = _determinism_case()

    gates = {
        "factorization_exact": bool(factorization["factorization_exact"]),
        "reference_oracle_representation_exact": bool(
            factorization["reference_oracle_representation_exact"]
        ),
        "flat_envelope_exact_identity": bool(identity["flat_envelope_exact_identity"]),
        "source_bypass_absent": bool(identity["source_bypass_absent"]),
        "exact_output_length": bool(identity["exact_output_length"])
        and bool(unvoiced_grid["exact_output_length"])
        and bool(voiced_grid["exact_output_length"]),
        "deterministic_same_seed": bool(determinism["deterministic_same_seed"]),
        "seed_changes_aperiodic_realization": bool(
            determinism["seed_changes_aperiodic_realization"]
        ),
        "unvoiced_grid_excess_rejected": not bool(unvoiced_grid["severe_grid_excess"]),
        "voiced_grid_excess_rejected": not bool(voiced_grid["severe_grid_excess"]),
    }
    status = "pass" if all(gates.values()) else "fail"
    return {
        "status": status,
        "audit_version": AUDIT_VERSION,
        "architecture_contract_version": ARCHITECTURE_CONTRACT_VERSION,
        "renderer_version": renderer.RENDERER_VERSION,
        "factorization": factorization,
        "identity_and_source_path": identity,
        "unvoiced_grid_safety": unvoiced_grid,
        "voiced_grid_safety": voiced_grid,
        "determinism": determinism,
        "gates": gates,
        "training_started": False,
        "optimizer_created": False,
        "model_instantiated": False,
        "checkpoints_touched": False,
        "persistent_training_authorized": False,
        "new_vocoder_checkpoint_authorized": False,
        "next_gate": (
            "review_owned_minimum_phase_renderer_safety_before_authorizing_model_instantiation"
            if status == "pass"
            else "revise_owned_minimum_phase_renderer_before_model_instantiation"
        ),
    }


def main() -> None:
    print(json.dumps(run_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
