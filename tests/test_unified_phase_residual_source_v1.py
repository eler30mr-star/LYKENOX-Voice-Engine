from __future__ import annotations

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_unified_phase_residual_source_v1 import (
    HARMONIC_COUNT,
    RESIDUAL_VECTOR_SAMPLES,
    LykenoxUnifiedPhaseResidualSourceV1,
)
from lykenox_voice_engine.training.speech_vocoder_unified_phase_residual_source_train_v1 import (
    synthesize_unified_residual,
)


def _inputs(frames: int = 8):
    mel = torch.zeros(1, frames, 80)
    f0 = torch.full((1, frames), 140.0)
    voiced = torch.ones(1, frames)
    periodicity = torch.full((1, frames), 0.8)
    return mel, f0, voiced, periodicity


def test_model_has_one_state_and_expected_output_geometry():
    model = LykenoxUnifiedPhaseResidualSourceV1().cpu()
    harmonic, periodic_level, aperiodic, aperiodic_level = model(*_inputs())
    assert harmonic.shape == (1, 8, HARMONIC_COUNT, 2)
    assert periodic_level.shape == (1, 8)
    assert aperiodic.shape == (1, 9, RESIDUAL_VECTOR_SAMPLES)
    assert aperiodic_level.shape == (1, 9)
    assert model.recurrent.bidirectional is False
    assert model.recurrent.num_layers == 1


def test_explicit_periodic_and_aperiodic_levels_match_generated_coordinates():
    model = LykenoxUnifiedPhaseResidualSourceV1().cpu()
    harmonic, periodic_level, aperiodic, aperiodic_level = model(*_inputs())
    fourier_rms = torch.sqrt(0.5 * harmonic.square().sum(dim=(-1, -2)).clamp_min(1.0e-12))
    vector_rms = torch.sqrt(aperiodic.square().mean(dim=-1).clamp_min(1.0e-12))
    assert torch.allclose(fourier_rms, torch.exp(periodic_level), atol=2.0e-4, rtol=2.0e-4)
    assert torch.allclose(vector_rms, torch.exp(aperiodic_level), atol=2.0e-4, rtol=2.0e-4)


def test_complementary_energy_endpoints_have_single_coordinate_authority():
    model = LykenoxUnifiedPhaseResidualSourceV1().cpu()
    mel, f0, voiced, _ = _inputs(frames=6)
    harmonic, _, aperiodic, _ = model(mel, f0, voiced, torch.ones_like(voiced))

    periodic_residual, periodic_wave, _, periodic_strength = synthesize_unified_residual(
        harmonic, aperiodic, f0, voiced, torch.ones_like(voiced)
    )
    assert torch.all(periodic_strength == 1.0)
    assert torch.allclose(periodic_residual, periodic_wave, atol=1.0e-6, rtol=1.0e-6)

    aperiodic_residual, _, aperiodic_wave, aperiodic_strength = synthesize_unified_residual(
        harmonic, aperiodic, f0, voiced, torch.zeros_like(voiced)
    )
    assert torch.all(aperiodic_strength == 0.0)
    assert torch.allclose(aperiodic_residual, aperiodic_wave, atol=1.0e-6, rtol=1.0e-6)


def test_transition_is_internal_energy_partition_not_source_checkpoint_handoff():
    model = LykenoxUnifiedPhaseResidualSourceV1().cpu()
    mel, f0, voiced, _ = _inputs(frames=7)
    periodicity = torch.linspace(1.0, 0.0, 7).unsqueeze(0)
    harmonic, _, aperiodic, _ = model(mel, f0, voiced, periodicity)
    residual, periodic_wave, aperiodic_wave, strength = synthesize_unified_residual(
        harmonic, aperiodic, f0, voiced, periodicity
    )
    expected = periodic_wave * torch.sqrt(strength) + aperiodic_wave * torch.sqrt(1.0 - strength)
    assert torch.allclose(residual, expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.isfinite(residual).all()
