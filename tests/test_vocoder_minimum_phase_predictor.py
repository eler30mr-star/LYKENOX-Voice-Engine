from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_v1 import (
    CEPSTRAL_ORDER,
    MEL_BINS,
    PREDICTOR_ARCHITECTURE,
    LykenoxFrameRateCepstralPredictorV1,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_predictor_smoke as smoke


class VocoderMinimumPhasePredictorTests(unittest.TestCase):
    def test_predictor_is_frame_rate_and_neutral_at_initialization(self) -> None:
        torch.manual_seed(11)
        model = LykenoxFrameRateCepstralPredictorV1()
        mel = torch.randn(2, 17, MEL_BINS)
        f0 = torch.full((2, 17), 190.0)
        voiced = torch.ones(2, 17)
        periodicity = torch.full((2, 17), 0.85)
        cepstrum = model(mel, f0, voiced, periodicity)
        self.assertEqual(PREDICTOR_ARCHITECTURE, "lykenox_owned_frame_rate_cepstral_predictor_v1")
        self.assertEqual(tuple(cepstrum.shape), (2, 17, CEPSTRAL_ORDER))
        self.assertEqual(int(torch.count_nonzero(cepstrum)), 0)
        self.assertEqual(float(model.cepstral_projection.weight.abs().max()), 0.0)
        self.assertEqual(float(model.cepstral_projection.bias.abs().max()), 0.0)

    def test_conditioning_features_preserve_exact_frame_count(self) -> None:
        model = LykenoxFrameRateCepstralPredictorV1()
        mel = torch.zeros(1, 9, MEL_BINS)
        f0 = torch.linspace(0.0, 240.0, 9).unsqueeze(0)
        voiced = torch.linspace(0.0, 1.0, 9).unsqueeze(0)
        periodicity = torch.linspace(1.0, 0.0, 9).unsqueeze(0)
        features = model.conditioning_features(mel, f0, voiced, periodicity)
        self.assertEqual(tuple(features.shape), (1, 9, MEL_BINS + 3))
        self.assertEqual(tuple(model(mel, f0, voiced, periodicity).shape), (1, 9, CEPSTRAL_ORDER))

    def test_predictor_contains_no_historical_waveform_failure_mechanism(self) -> None:
        source = inspect.getsource(
            __import__(
                "lykenox_voice_engine.models.vocoder.network_minimum_phase_v1",
                fromlist=["*"],
            )
        ).lower()
        for forbidden in (
            "convtranspose",
            "conv_transpose",
            "torch.stft",
            "torch.istft",
            "torch.fft",
            "interpolate(",
            "torch.optim",
            "from_pretrained",
            "waveform_head",
            "harmonic_sinusoid_bank",
        ):
            self.assertNotIn(forbidden, source)

    def test_structural_smoke_passes_without_optimizer_or_update(self) -> None:
        result = smoke.run_smoke()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["model_instantiated"])
        self.assertFalse(result["optimizer_created"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["parameter_update_executed"])
        self.assertFalse(result["checkpoint_loaded"])
        self.assertFalse(result["checkpoint_saved"])
        self.assertFalse(result["persistent_training_authorized"])
        self.assertFalse(result["new_vocoder_checkpoint_authorized"])
        for value in result["gates"].values():
            self.assertTrue(value)

    def test_smoke_source_cannot_train_or_write_checkpoint(self) -> None:
        source = inspect.getsource(smoke).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            ".step(",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "vocos",
            "bigvgan",
            "hifigan",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn("parameter_update_executed\": false", source)


if __name__ == "__main__":
    unittest.main()
