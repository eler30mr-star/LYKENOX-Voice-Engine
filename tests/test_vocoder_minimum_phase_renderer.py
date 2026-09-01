from __future__ import annotations

import inspect
import math
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer


class VocoderMinimumPhaseRendererTests(unittest.TestCase):
    def test_renderer_contract_constants_are_exact(self) -> None:
        self.assertEqual(renderer.RENDERER_VERSION, "owned-minimum-phase-time-varying-renderer-v1")
        self.assertEqual(renderer.SAMPLE_RATE, 24000)
        self.assertEqual(renderer.HOP_LENGTH, 256)
        self.assertEqual(renderer.N_FFT, 1024)
        self.assertEqual(renderer.CEPSTRAL_ORDER, 64)

    def test_cepstrum_to_minimum_phase_fir_preserves_log_magnitude(self) -> None:
        cepstrum = torch.zeros(2, renderer.CEPSTRAL_ORDER, dtype=torch.float64)
        cepstrum[0, 0] = 0.10
        cepstrum[0, 1] = 0.05
        cepstrum[0, 2] = -0.03
        cepstrum[0, 5] = 0.02
        cepstrum[1, 0] = -0.20
        cepstrum[1, 3] = 0.04
        cepstrum[1, 10] = -0.01

        impulse = renderer.one_sided_real_cepstrum_to_minimum_phase_fir(cepstrum)
        represented = renderer.cepstral_log_magnitude(cepstrum)
        reconstructed = torch.log(
            torch.fft.rfft(impulse, n=renderer.N_FFT, dim=-1).abs().clamp_min(1e-30)
        )
        self.assertLess(float((represented - reconstructed).abs().max()), 1e-10)

    def test_reference_log_magnitude_oracle_roundtrips_contract_representation(self) -> None:
        cepstrum = torch.zeros(3, renderer.CEPSTRAL_ORDER, dtype=torch.float64)
        cepstrum[:, 0] = torch.tensor([0.0, 0.12, -0.08], dtype=torch.float64)
        cepstrum[:, 1] = torch.tensor([0.05, -0.03, 0.02], dtype=torch.float64)
        cepstrum[:, 4] = torch.tensor([-0.02, 0.01, 0.03], dtype=torch.float64)
        log_magnitude = renderer.cepstral_log_magnitude(cepstrum)
        recovered = renderer.reference_log_magnitude_to_one_sided_cepstrum(log_magnitude)
        self.assertLess(float((recovered - cepstrum).abs().max()), 1e-10)

    def test_flat_envelope_is_exact_renderer_identity(self) -> None:
        frame_count = 7
        sample_count = frame_count * renderer.HOP_LENGTH
        index = torch.arange(sample_count, dtype=torch.float64)
        excitation = (
            torch.sin(2.0 * math.pi * 173.0 * index / renderer.SAMPLE_RATE)
            + 0.2 * torch.sin(2.0 * math.pi * 613.0 * index / renderer.SAMPLE_RATE)
        ).unsqueeze(0)
        cepstrum = torch.zeros(
            1,
            frame_count,
            renderer.CEPSTRAL_ORDER,
            dtype=torch.float64,
        )
        waveform = renderer.render_time_varying_minimum_phase(excitation, cepstrum)
        self.assertEqual(tuple(waveform.shape), (1, sample_count))
        self.assertEqual(float((waveform - excitation).abs().max()), 0.0)

    def test_dc_cepstrum_attenuation_proves_no_source_bypass(self) -> None:
        frame_count = 6
        sample_count = frame_count * renderer.HOP_LENGTH
        index = torch.arange(sample_count, dtype=torch.float64)
        excitation = (
            torch.sin(2.0 * math.pi * 197.0 * index / renderer.SAMPLE_RATE)
            + 0.17 * torch.cos(2.0 * math.pi * 881.0 * index / renderer.SAMPLE_RATE)
        ).unsqueeze(0)
        attenuation_log = -6.0
        cepstrum = torch.zeros(
            1,
            frame_count,
            renderer.CEPSTRAL_ORDER,
            dtype=torch.float64,
        )
        cepstrum[..., 0] = attenuation_log
        waveform = renderer.render_time_varying_minimum_phase(excitation, cepstrum)
        expected = excitation * math.exp(attenuation_log)
        self.assertLess(float((waveform - expected).abs().max()), 1e-12)
        self.assertLess(
            float(waveform.square().mean().sqrt() / excitation.square().mean().sqrt()),
            0.003,
        )

    def test_fixed_interpolation_has_exact_sample_clock(self) -> None:
        frames = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float64)
        samples = renderer.fixed_linear_frame_to_sample(frames)
        self.assertEqual(samples.shape[-1], 3 * renderer.HOP_LENGTH)
        self.assertEqual(float(samples[0, 0]), 0.0)
        self.assertEqual(float(samples[0, renderer.HOP_LENGTH]), 1.0)
        self.assertEqual(float(samples[0, 2 * renderer.HOP_LENGTH]), -1.0)
        self.assertEqual(float(samples[0, -1]), -1.0)

    def test_neutral_excitation_is_deterministic_and_exact_length(self) -> None:
        frame_count = 9
        f0 = torch.zeros(1, frame_count, dtype=torch.float64)
        voiced = torch.zeros_like(f0)
        periodicity = torch.zeros_like(f0)
        first = renderer.build_neutral_excitation(f0, voiced, periodicity, noise_seed=17)
        second = renderer.build_neutral_excitation(f0, voiced, periodicity, noise_seed=17)
        different = renderer.build_neutral_excitation(f0, voiced, periodicity, noise_seed=18)
        self.assertEqual(first.shape[-1], frame_count * renderer.HOP_LENGTH)
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(float((first - different).abs().max()), 1e-3)

    def test_full_fixed_path_exact_length_and_finite_for_voiced_conditioning(self) -> None:
        frame_count = 12
        frame_index = torch.arange(frame_count, dtype=torch.float64)
        f0 = (185.0 + 18.0 * torch.sin(frame_index * 0.31)).unsqueeze(0)
        voiced = torch.ones_like(f0)
        periodicity = (0.80 + 0.08 * torch.sin(frame_index * 0.17)).unsqueeze(0)
        cepstrum = torch.zeros(
            1,
            frame_count,
            renderer.CEPSTRAL_ORDER,
            dtype=torch.float64,
        )
        cepstrum[0, :, 1] = 0.05 * torch.sin(frame_index * 0.73)
        cepstrum[0, :, 2] = 0.03 * torch.cos(frame_index * 1.11)
        waveform, excitation = renderer.render_owned_minimum_phase_vocoder_path(
            cepstrum,
            f0,
            voiced,
            periodicity,
            noise_seed=5,
        )
        expected_samples = frame_count * renderer.HOP_LENGTH
        self.assertEqual(tuple(waveform.shape), (1, expected_samples))
        self.assertEqual(tuple(excitation.shape), (1, expected_samples))
        self.assertTrue(torch.isfinite(waveform).all())
        self.assertTrue(torch.isfinite(excitation).all())

    def test_renderer_contains_no_model_optimizer_checkpoint_or_training_path(self) -> None:
        source = inspect.getsource(renderer).lower()
        for forbidden in (
            "nn.module",
            "torch.optim",
            ".backward(",
            ".step(",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "convtranspose",
            "vocos",
            "bigvgan",
            "hifigan",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
