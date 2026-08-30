from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training.speech_vocoder_v7_full_utterance_oracle_probe import (
    AUDIT_VERSION,
    OUTPUT_DIR_NAME,
    VALIDATION_INDICES,
    V7_ARTIFACT_DIR_NAME,
    run_v7_full_utterance_oracle_probe,
)
from lykenox_voice_engine.training.speech_vocoder_v7_train import ARTIFACT_DIR_NAME


class VocoderV7OracleContractTests(unittest.TestCase):
    def test_oracle_targets_exact_first_epoch_artifact_and_three_full_utterances(self) -> None:
        self.assertEqual(AUDIT_VERSION, "vocoder-v7-epoch1-full-utterance-oracle-probe-v1")
        self.assertEqual(V7_ARTIFACT_DIR_NAME, ARTIFACT_DIR_NAME)
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        self.assertIn("epoch1", OUTPUT_DIR_NAME)

    def test_oracle_is_read_only_with_hard_perceptual_gate(self) -> None:
        source = inspect.getsource(run_v7_full_utterance_oracle_probe)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "run_bounded_resumable_v7_first_epoch(",
            "posthoc_gain_normalization_used\": True",
            "posthoc_eq_used\": True",
            "posthoc_denoising_used\": True",
            "epoch2_training_authorized\": True",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"epoch2_training_authorized": False', source)
        self.assertIn('"full_utterance_perceptual_acceptance": False', source)
        self.assertIn('"listening_order": "reference -> v4.2 -> v7 epoch1"', source)
        self.assertIn("checkpoints_unchanged = before == after", source)


if __name__ == "__main__":
    unittest.main()
