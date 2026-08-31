from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_pretrained_vocos_copy_synthesis as probe
from lykenox_voice_engine.training.speech_vocoder_v4_2_replacement_decision import (
    ACOUSTIC_TRAINING_AUTHORIZED,
    NEXT_ARCHITECTURE,
    NEXT_GATE,
    SCRATCH_VOCODER_ITERATION_AUTHORIZED,
    V4_2_FURTHER_TRAINING_AUTHORIZED,
)
from lykenox_voice_engine.training.speech_vocoder_v9_rejection import (
    V9_PERCEPTUALLY_REJECTED,
    V9_TRAINING_ENABLED,
)


class PretrainedVocosCopySynthesisContractTests(unittest.TestCase):
    def test_probe_identity_and_full_utterance_contract(self) -> None:
        self.assertEqual(
            probe.PROBE_VERSION,
            "pretrained-vocos-24khz-full-utterance-copy-synthesis-v1",
        )
        self.assertEqual(probe.MODEL_ID, "charactr/vocos-mel-24khz")
        self.assertEqual(probe.SAMPLE_RATE, 24000)
        self.assertEqual(probe.HOP_LENGTH, 256)
        self.assertEqual(probe.VALIDATION_INDICES, (0, 1, 2))

    def test_probe_is_read_only_for_lykenox_models(self) -> None:
        source = inspect.getsource(probe)
        run_source = inspect.getsource(probe.run_pretrained_vocos_copy_synthesis)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("load_v4_2_checkpoint", source)
        self.assertNotIn("load_v6_checkpoint", source)
        self.assertNotIn("load_v7", source)
        self.assertNotIn("load_v9", source)
        self.assertNotIn("optimizer", source.lower())
        self.assertIn('"training_started": False', run_source)
        self.assertIn('"persistent_training_authorized": False', run_source)
        self.assertIn('"predicted_duration_modified": False', run_source)
        self.assertIn('"posthoc_gain_normalization_used": False', run_source)
        self.assertIn('"posthoc_eq_used": False', run_source)
        self.assertIn('"posthoc_denoising_used": False', run_source)
        self.assertIn('"full_utterance": True', run_source)
        self.assertIn('"needs_listening"', run_source)

    def test_identical_pair_metrics_are_neutral(self) -> None:
        torch.manual_seed(501)
        wave = torch.randn(4096) * 0.02
        metrics = probe._paired_metrics(wave, wave.clone())
        self.assertLess(abs(metrics["rms_relative_db"]), 1e-7)
        self.assertLess(abs(metrics["spectral_centroid_relative_pct"]), 1e-7)
        self.assertLess(abs(metrics["presence_1k_8k_error_db"]), 1e-7)

    def test_scratch_chain_is_closed_before_probe(self) -> None:
        self.assertTrue(V9_PERCEPTUALLY_REJECTED)
        self.assertFalse(V9_TRAINING_ENABLED)
        self.assertFalse(V4_2_FURTHER_TRAINING_AUTHORIZED)
        self.assertFalse(ACOUSTIC_TRAINING_AUTHORIZED)
        self.assertFalse(SCRATCH_VOCODER_ITERATION_AUTHORIZED)
        self.assertEqual(NEXT_ARCHITECTURE, "pretrained_vocoder_baseline")
        self.assertEqual(
            NEXT_GATE,
            "run_full_utterance_pretrained_vocos_copy_synthesis_before_any_more_vocoder_training",
        )


if __name__ == "__main__":
    unittest.main()
