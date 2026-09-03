from __future__ import annotations

import math

import torch

from scripts.render_pitch_synchronous_cycle_source_v2_phase_continuous import (
    decode_cycle_periodic_bandlimited,
    synthesize_phase_continuous_cycles,
)


def test_periodic_decoder_preserves_constant_level():
    canonical = torch.full((128,), 0.125, dtype=torch.float32)
    for samples in (25, 64, 127, 193):
        decoded = decode_cycle_periodic_bandlimited(canonical, samples)
        assert decoded.shape == (samples,)
        assert torch.allclose(decoded, torch.full_like(decoded, 0.125), atol=1.0e-6, rtol=1.0e-6)


def test_periodic_decoder_tracks_safe_harmonic_without_linear_resampling_images():
    phase = torch.arange(128, dtype=torch.float32) / 128.0
    canonical = torch.sin(2.0 * math.pi * 3.0 * phase)
    decoded = decode_cycle_periodic_bandlimited(canonical, 97)
    expected_phase = torch.arange(97, dtype=torch.float32) / 97.0
    expected = torch.sin(2.0 * math.pi * 3.0 * expected_phase)
    assert torch.allclose(decoded, expected, atol=2.0e-5, rtol=2.0e-5)


def test_cycle_morph_approaches_next_cycle_at_boundary():
    phase = torch.arange(128, dtype=torch.float32) / 128.0
    first = torch.sin(2.0 * math.pi * phase)
    second = 0.5 * torch.sin(2.0 * math.pi * phase + 0.3)
    cycles = torch.stack((first, second), dim=0)
    boundaries = [(10, 110, 0, 1.0), (110, 210, 1, 1.0)]
    waveform, coverage = synthesize_phase_continuous_cycles(cycles, boundaries, output_samples=220)
    assert waveform.shape == (220,)
    assert coverage[10:210].min().item() == 1.0
    # The boundary jump must be small relative to the waveform's ordinary sample-to-sample slope.
    boundary_jump = (waveform[110] - waveform[109]).abs()
    ordinary_slope = (waveform[11:109] - waveform[10:108]).abs().mean().clamp_min(1.0e-6)
    assert float(boundary_jump / ordinary_slope) < 4.0


def test_high_harmonics_above_physical_cycle_nyquist_are_removed():
    phase = torch.arange(128, dtype=torch.float32) / 128.0
    canonical = torch.sin(2.0 * math.pi * 30.0 * phase)
    decoded = decode_cycle_periodic_bandlimited(canonical, 24)
    # A 24-sample physical period can represent at most harmonic 11 safely under this decoder.
    assert float(decoded.abs().max()) < 1.0e-4
