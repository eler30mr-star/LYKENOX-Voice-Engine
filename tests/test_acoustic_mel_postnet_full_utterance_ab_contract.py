from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training.speech_acoustic_mel_postnet_full_utterance_ab import (
    AUDIT_VERSION,
    OUTPUT_DIR_NAME,
    VALIDATION_INDICES,
    VARIANTS,
    run_mel_postnet_full_utterance_ab,
)


class AcousticMelPostnetFullUtteranceABContractTests(unittest.TestCase):
    def test_identity_and_variants_are_fixed(self) -> None:
        self.assertEqual(
            AUDIT_VERSION,
            "acoustic-mel-postnet-full-utterance-v4-2-ab-v1",
        )
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        self.assertEqual(
            VARIANTS,
            (
                "v4_2_oracle",
                "base_mel_target_prosody",
                "postnet_mel_target_prosody",
                "base_mel_predicted_prosody",
                "postnet_mel_predicted_prosody",
            ),
        )
        self.assertIn("full_utterance", OUTPUT_DIR_NAME)

    def test_gate_is_read_only_and_keeps_epoch2_closed(self) -> None:
        source = inspect.getsource(run_mel_postnet_full_utterance_ab)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "run_mel_postnet_training",
            '"epoch2_training_authorized": True',
            '"posthoc_gain_normalization_used": True',
            '"posthoc_eq_used": True',
            '"posthoc_denoising_used": True',
            '"predicted_duration_modified": True',
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"epoch2_training_authorized": False', source)
        self.assertIn('"training_started": False', source)
        self.assertIn('"predicted_duration_modified": False', source)
        self.assertIn("checkpoints_unchanged", source)

    def test_gate_requires_exact_non_mel_outputs_and_base_mel_identity(self) -> None:
        source = inspect.getsource(run_mel_postnet_full_utterance_ab)
        for key in (
            "duration_prediction",
            "regulated_durations",
            "f0_prediction_hz",
            "voicing_logits",
            "mel_lengths",
            "mel_mask",
        ):
            self.assertIn(f'"{key}"', source)
        self.assertIn('postnet_output["base_mel"]', source)
        self.assertIn("predicted_prosody_exact", source)
        self.assertIn("prepare_speech_vocoder_conditioning", source)

    def test_postnet_is_perceptually_rejected_and_epoch_two_stays_closed(self) -> None:
        source = inspect.getsource(run_mel_postnet_full_utterance_ab)
        self.assertIn("perceptually rejected", source)
        self.assertIn("slightly below the accepted v4.2 baseline", source)
        self.assertIn(
            '"next_gate": "postnet_perceptually_rejected_no_epoch2"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
