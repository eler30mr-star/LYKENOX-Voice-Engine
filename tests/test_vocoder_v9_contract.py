from __future__ import annotations

import inspect
import unittest

import torch

import lykenox_voice_engine.models.vocoder.network_v9 as network_v9
from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV9,
    VOCODER_GENERATOR_V9_ARCHITECTURE,
)
from lykenox_voice_engine.training import speech_vocoder_v9_architecture_smoke as v9_smoke
from lykenox_voice_engine.training.speech_vocoder_v4_2_replacement_decision import (
    NEXT_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_vocoder_v9_phase_increment_loss import (
    V9_PHASE_INCREMENT_LOSS_VERSION,
    v9_phase_increment_loss,
)


class V9VocoderContractTests(unittest.TestCase):
    def _model(self) -> LykenoxVocoderGeneratorV9:
        return LykenoxVocoderGeneratorV9(hidden_channels=64).cpu().eval()

    def test_factor_shapes_and_exact_waveform_length(self) -> None:
        torch.manual_seed(91)
        model = self._model()
        frames = 6
        mel = torch.randn(2, frames, model.config.mel_bins)
        f0 = torch.full((2, frames), 125.0)
        voiced = torch.ones(2, frames)
        with torch.inference_mode():
            magnitude, residual = model.predict_spectral_factors(mel, f0, voiced)
            spectrum = model.spectrum_from_factors(magnitude, residual)
            waveform = model(mel, f0, voiced)
        expected = (2, model.frequency_bins, frames + 1)
        self.assertEqual(tuple(magnitude.shape), expected)
        self.assertEqual(tuple(residual.shape), expected)
        self.assertEqual(tuple(spectrum.shape), expected)
        self.assertTrue(torch.is_complex(residual))
        self.assertTrue(torch.is_complex(spectrum))
        self.assertLess(float((residual.abs() - 1.0).abs().max()), 1e-5)
        self.assertEqual(tuple(waveform.shape), (2, frames * model.config.hop_length))

    def test_target_factorization_reconstructs_complex_spectrum_and_waveform(self) -> None:
        torch.manual_seed(92)
        model = self._model()
        samples = 12 * model.config.hop_length
        target = torch.randn(1, samples) * 0.05
        with torch.inference_mode():
            spectrum = model.target_complex_spectrum(target)
            magnitude, residual = model.factorize_target_spectrum(spectrum)
            reconstructed_spectrum = model.spectrum_from_factors(magnitude, residual)
            reconstructed_wave = model.synthesize_complex_spectrum(
                reconstructed_spectrum,
                samples=samples,
            )
        self.assertLess(float((reconstructed_spectrum - spectrum).abs().mean()), 1e-4)
        self.assertLess(float((reconstructed_wave - target).abs().mean()), 1e-5)

    def test_zero_residual_increment_uses_stft_bin_advance_not_phase_reset(self) -> None:
        model = self._model()
        frames = 5
        residual = torch.ones(
            1,
            model.frequency_bins,
            frames,
            dtype=torch.complex64,
        )
        with torch.inference_mode():
            phase = model.integrate_phase_residual(residual)
        observed_step = phase[..., 1:] * phase[..., :-1].conj()
        expected_step = model.bin_phase_advance.view(1, -1, 1).expand_as(observed_step)
        self.assertLess(float((observed_step - expected_step).abs().max()), 1e-5)
        self.assertFalse(model.absolute_frame_phase_prediction)

    def test_f0_and_voicing_are_conditioning_only(self) -> None:
        torch.manual_seed(93)
        model = self._model()
        frames = 5
        mel = torch.randn(1, frames, model.config.mel_bins)
        low_f0 = torch.full((1, frames), 90.0)
        high_f0 = torch.full((1, frames), 210.0)
        voiced = torch.ones(1, frames)
        unvoiced = torch.zeros(1, frames)
        with torch.inference_mode():
            low_mag, _low_phase = model.predict_spectral_factors(mel, low_f0, voiced)
            high_mag, _high_phase = model.predict_spectral_factors(mel, high_f0, voiced)
            uv_mag, _uv_phase = model.predict_spectral_factors(mel, low_f0, unvoiced)
        self.assertFalse(torch.equal(low_mag, high_mag))
        self.assertFalse(torch.equal(low_mag, uv_mag))
        self.assertTrue(model.source_free)
        self.assertFalse(model.explicit_sample_rate_source)
        self.assertFalse(model.learned_sample_rate_upsampling)

    def test_architecture_excludes_prior_source_and_upsampling_failures(self) -> None:
        source = inspect.getsource(network_v9)
        self.assertEqual(
            VOCODER_GENERATOR_V9_ARCHITECTURE,
            "lykenox_phase_increment_spectral_ola_v9",
        )
        self.assertIn("torch.cumprod(", source)
        self.assertIn("bin_phase_advance", source)
        self.assertIn("factorize_target_spectrum", source)
        self.assertIn("torch.istft(", source)
        for forbidden in (
            "nn.ConvTranspose1d(",
            "F.interpolate(",
            "def _harmonic_source(",
            "def _aperiodic_source(",
            "self.source_gate =",
            "torch.randn_like(",
        ):
            self.assertNotIn(forbidden, source)

    def test_phase_increment_loss_is_phase_sensitive_and_differentiable(self) -> None:
        self.assertEqual(
            V9_PHASE_INCREMENT_LOSS_VERSION,
            "vocoder-v9-phase-increment-loss-v1",
        )
        torch.manual_seed(94)
        magnitude = torch.rand(1, 9, 4) + 0.1
        predicted_magnitude = magnitude.clone().requires_grad_(True)
        target_phase = torch.polar(torch.ones(1, 9, 4), torch.randn(1, 9, 4))
        phase_angle = torch.randn(1, 9, 4, requires_grad=True)
        predicted_phase = torch.polar(torch.ones_like(phase_angle), phase_angle)
        target_wave = torch.randn(1, 256)
        predicted_wave = target_wave.clone().requires_grad_(True)
        result = v9_phase_increment_loss(
            predicted_magnitude,
            magnitude,
            predicted_phase,
            target_phase,
            predicted_wave,
            target_wave,
        )
        self.assertGreater(float(result.phase_increment_circular.detach()), 0.0)
        result.total.backward()
        self.assertIsNotNone(predicted_magnitude.grad)
        self.assertIsNotNone(phase_angle.grad)

    def test_smoke_is_bounded_reference_relative_and_nonpersistent(self) -> None:
        source = inspect.getsource(v9_smoke.run_v9_architecture_smoke)
        module_source = inspect.getsource(v9_smoke)
        self.assertEqual(v9_smoke.SMOKE_VERSION, "vocoder-v9-phase-increment-ola-smoke-v1")
        self.assertIn("factorized_spectrum_mae", source)
        self.assertIn("factorized_waveform_mae", source)
        self.assertIn("phase_increment_decreased", source)
        self.assertIn("frame_grid_artifact_excess_metrics", module_source)
        self.assertIn('"persistent_training_started": False', source)
        self.assertIn('"persistent_training_authorized": False', source)
        self.assertNotIn("torch.save(", module_source)
        self.assertEqual(NEXT_ARCHITECTURE, VOCODER_GENERATOR_V9_ARCHITECTURE)


if __name__ == "__main__":
    unittest.main()
