from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderGeneratorV8,
    VOCODER_GENERATOR_V8_ARCHITECTURE,
    VOCODER_GENERATOR_V9_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    VOCODER_GRID_ARTIFACT_EXCESS_VERSION,
    frame_grid_artifact_excess_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_replacement_decision import (
    ACOUSTIC_TRAINING_AUTHORIZED,
    NEXT_ARCHITECTURE,
    V4_2_FURTHER_TRAINING_AUTHORIZED,
    V4_2_ROLE,
    V8_VERDICT,
)
from lykenox_voice_engine.training.speech_vocoder_v8_rejection import (
    V8_ARCHITECTURALLY_REJECTED,
    V8_REJECTION_GATE,
    V8_TRAINING_ENABLED,
    require_v8_training_enabled,
)


class V8VocoderContractTests(unittest.TestCase):
    def test_v8_fixed_renderer_remains_forensically_reproducible(self) -> None:
        torch.manual_seed(81)
        model = LykenoxVocoderGeneratorV8(hidden_channels=64).cpu().eval()
        self.assertEqual(VOCODER_GENERATOR_V8_ARCHITECTURE, "lykenox_complex_spectral_overlap_add_v8")
        samples = 10 * model.config.hop_length
        target = torch.randn(1, samples) * 0.05
        with torch.inference_mode():
            spectrum = model.target_complex_spectrum(target)
            reconstructed = model.synthesize_complex_spectrum(spectrum, samples=samples)
        self.assertEqual(tuple(reconstructed.shape), tuple(target.shape))
        self.assertLess(float((reconstructed - target).abs().mean()), 1e-5)

    def test_reference_relative_grid_gate_preserves_identical_periodic_audio(self) -> None:
        self.assertEqual(
            VOCODER_GRID_ARTIFACT_EXCESS_VERSION,
            "vocoder-frame-grid-artifact-excess-v1",
        )
        torch.manual_seed(82)
        frame = torch.randn(256)
        reference = frame.repeat(8).unsqueeze(0)
        result = frame_grid_artifact_excess_metrics(
            reference.clone(),
            reference,
            sample_rate=24000,
            hop_length=256,
        )
        self.assertFalse(bool(result.severe_grid_excess[0]))
        self.assertLess(abs(float(result.hop_autocorrelation_excess[0])), 1e-7)

    def test_v8_is_permanently_blocked_after_valid_v2_smoke(self) -> None:
        self.assertTrue(V8_ARCHITECTURALLY_REJECTED)
        self.assertFalse(V8_TRAINING_ENABLED)
        self.assertEqual(V8_REJECTION_GATE, "reference_relative_frame_grid_architecture_smoke_v2")
        with self.assertRaises(RuntimeError):
            require_v8_training_enabled()
        self.assertIn("hop-locked", V8_VERDICT)

    def test_replacement_decision_advances_to_v9_without_reopening_acoustic(self) -> None:
        self.assertEqual(V4_2_ROLE, "intelligible_colored_baseline_only")
        self.assertFalse(V4_2_FURTHER_TRAINING_AUTHORIZED)
        self.assertFalse(ACOUSTIC_TRAINING_AUTHORIZED)
        self.assertEqual(NEXT_ARCHITECTURE, VOCODER_GENERATOR_V9_ARCHITECTURE)


if __name__ == "__main__":
    unittest.main()
