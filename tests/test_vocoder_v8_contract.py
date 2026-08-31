from __future__ import annotations

import inspect
import unittest

import torch

import lykenox_voice_engine.models.vocoder.network_v8 as network_v8
from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV8,
    VOCODER_GENERATOR_V8_ARCHITECTURE,
)
from lykenox_voice_engine.training import speech_vocoder_v8_architecture_smoke as v8_smoke
from lykenox_voice_engine.training.speech_vocoder_v4_2_replacement_decision import (
    ACOUSTIC_TRAINING_AUTHORIZED,
    NEXT_ARCHITECTURE,
    V4_2_FURTHER_TRAINING_AUTHORIZED,
    V4_2_ROLE,
)
from lykenox_voice_engine.training.speech_vocoder_v8_complex_spectral_loss import (
    V8_COMPLEX_SPECTRAL_LOSS_VERSION,
    v8_complex_spectral_loss,
)


class V8VocoderContractTests(unittest.TestCase):
    def _model(self) -> LykenoxVocoderGeneratorV8:
        return LykenoxVocoderGeneratorV8(hidden_channels=64).cpu().eval()

    def test_complex_spectrum_and_exact_waveform_length(self) -> None:
        torch.manual_seed(81)
        model = self._model()
        frames = 6
        mel = torch.randn(2, frames, model.config.mel_bins)
        f0 = torch.full((2, frames), 125.0)
        voiced = torch.ones(2, frames)
        with torch.inference_mode():
            spectrum = model.predict_complex_spectrum(mel, f0, voiced)
            waveform = model(mel, f0, voiced)
        self.assertEqual(
            tuple(spectrum.shape),
            (2, model.frequency_bins, frames + 1),
        )
        self.assertTrue(torch.is_complex(spectrum))
        self.assertTrue(torch.isfinite(torch.view_as_real(spectrum)).all())
        self.assertEqual(
            tuple(waveform.shape),
            (2, frames * model.config.hop_length),
        )
        self.assertTrue(torch.isfinite(waveform).all())

    def test_fixed_stft_istft_roundtrip_is_near_exact(self) -> None:
        torch.manual_seed(82)
        model = self._model()
        samples = 10 * model.config.hop_length
        target = torch.randn(1, samples) * 0.05
        with torch.inference_mode():
            spectrum = model.target_complex_spectrum(target)
            reconstructed = model.synthesize_complex_spectrum(spectrum, samples=samples)
        self.assertEqual(
            tuple(spectrum.shape),
            (1, model.frequency_bins, 11),
        )
        self.assertEqual(tuple(reconstructed.shape), tuple(target.shape))
        self.assertLess(float((reconstructed - target).abs().mean()), 1e-5)

    def test_f0_and_voicing_are_frame_conditioning_not_waveform_sources(self) -> None:
        torch.manual_seed(83)
        model = self._model()
        frames = 5
        mel = torch.randn(1, frames, model.config.mel_bins)
        low_f0 = torch.full((1, frames), 90.0)
        high_f0 = torch.full((1, frames), 210.0)
        unvoiced = torch.zeros(1, frames)
        voiced = torch.ones(1, frames)
        with torch.inference_mode():
            low = model.predict_complex_spectrum(mel, low_f0, voiced)
            high = model.predict_complex_spectrum(mel, high_f0, voiced)
            uv = model.predict_complex_spectrum(mel, low_f0, unvoiced)
        self.assertFalse(torch.equal(low, high))
        self.assertFalse(torch.equal(low, uv))
        self.assertTrue(model.source_free)
        self.assertFalse(model.explicit_sample_rate_source)
        self.assertFalse(model.learned_sample_rate_upsampling)
        self.assertEqual(model.synthesis, "fixed_hann_istft_overlap_add")

    def test_architecture_excludes_prior_failure_mechanisms(self) -> None:
        source = inspect.getsource(network_v8)
        self.assertEqual(
            VOCODER_GENERATOR_V8_ARCHITECTURE,
            "lykenox_complex_spectral_overlap_add_v8",
        )
        self.assertIn("torch.istft(", source)
        self.assertIn("torch.stft(", source)
        self.assertIn("predict_complex_spectrum", source)
        for forbidden in (
            "nn.ConvTranspose1d(",
            "F.interpolate(",
            "torch.sin(",
            "torch.cumsum(",
            "def _harmonic_source(",
            "def _aperiodic_source(",
            "baseline_harmonic_weights =",
            "self.source_gate =",
        ):
            self.assertNotIn(forbidden, source)

    def test_complex_loss_is_phase_sensitive_and_differentiable(self) -> None:
        self.assertEqual(
            V8_COMPLEX_SPECTRAL_LOSS_VERSION,
            "vocoder-v8-complex-spectral-loss-v1",
        )
        torch.manual_seed(84)
        target_real = torch.randn(1, 9, 4)
        target_imag = torch.randn(1, 9, 4)
        target_spectrum = torch.complex(target_real, target_imag)
        predicted_real = target_real.clone().requires_grad_(True)
        predicted_imag = (-target_imag).clone().requires_grad_(True)
        predicted_spectrum = torch.complex(predicted_real, predicted_imag)
        target_wave = torch.randn(1, 256)
        predicted_wave = target_wave.clone().requires_grad_(True)
        result = v8_complex_spectral_loss(
            predicted_spectrum,
            target_spectrum,
            predicted_wave,
            target_wave,
        )
        self.assertGreater(float(result.complex_relative_l1.detach()), 0.0)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(predicted_real.grad)
        self.assertIsNotNone(predicted_imag.grad)

    def test_smoke_has_v7_missing_grid_gate_and_no_persistence(self) -> None:
        source = inspect.getsource(v8_smoke.run_v8_architecture_smoke)
        module_source = inspect.getsource(v8_smoke)
        self.assertIn("frame_grid_artifact_metrics", module_source)
        self.assertIn("fixed_stft_istft_roundtrip_mae", source)
        self.assertIn("final_grid_failure", source)
        self.assertIn('"persistent_training_started": False', source)
        self.assertIn('"persistent_training_authorized": False', source)
        self.assertIn('"metrics_can_accept_voice_quality": False', source)
        self.assertIn('"audible_full_utterance_acceptance_required": True', source)
        self.assertNotIn("torch.save(", module_source)
        self.assertNotIn("load_v4_2_checkpoint", module_source)
        self.assertNotIn("load_v6_checkpoint", module_source)
        self.assertNotIn("load_v7", module_source)

    def test_v4_2_is_frozen_as_baseline_only(self) -> None:
        self.assertEqual(V4_2_ROLE, "intelligible_colored_baseline_only")
        self.assertFalse(V4_2_FURTHER_TRAINING_AUTHORIZED)
        self.assertFalse(ACOUSTIC_TRAINING_AUTHORIZED)
        self.assertEqual(NEXT_ARCHITECTURE, VOCODER_GENERATOR_V8_ARCHITECTURE)


if __name__ == "__main__":
    unittest.main()
