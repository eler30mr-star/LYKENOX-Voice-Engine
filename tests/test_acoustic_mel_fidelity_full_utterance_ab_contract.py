from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_full_utterance_ab import (
    AUDIT_VERSION,
    OUTPUT_DIR_NAME,
    VALIDATION_INDICES,
    VARIANTS,
    run_acoustic_mel_fidelity_full_utterance_ab,
)


class AcousticMelFidelityFullUtteranceABContractTests(unittest.TestCase):
    def test_identity_and_variants_are_fixed(self) -> None:
        self.assertEqual(
            AUDIT_VERSION,
            "acoustic-mel-fidelity-full-utterance-v4-2-ab-v1",
        )
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        self.assertEqual(
            VARIANTS,
            (
                "v4_2_oracle",
                "base_mel_target_prosody",
                "refined_mel_target_prosody",
                "base_mel_predicted_prosody",
                "refined_mel_predicted_prosody",
            ),
        )
        self.assertIn("full_utterance", OUTPUT_DIR_NAME)

    def test_gate_is_read_only_and_keeps_epoch2_closed(self) -> None:
        source = inspect.getsource(run_acoustic_mel_fidelity_full_utterance_ab)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "run_isolated_mel_fidelity_training",
            '"epoch2_training_authorized": True',
            '"posthoc_gain_normalization_used": True',
            '"posthoc_eq_used": True',
            '"posthoc_denoising_used": True',
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("require_frozen_state_exact(refined, base)", source)
        self.assertIn('torch.equal(base_output["duration_prediction"]', source)
        self.assertIn('torch.equal(base_output["f0_prediction_hz"]', source)
        self.assertIn('torch.equal(base_output["voicing_logits"]', source)
        self.assertIn('"checkpoints_unchanged": checkpoints_unchanged', source)
        self.assertIn('"training_started": False', source)
        self.assertIn('"epoch2_training_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
