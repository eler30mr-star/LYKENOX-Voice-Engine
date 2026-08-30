from __future__ import annotations

from pathlib import Path
import unittest

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV6
from lykenox_voice_engine.training.speech_vocoder_v6_architecture_smoke import (
    run_v6_architecture_smoke,
)
from lykenox_voice_engine.training.speech_vocoder_v6_clarity_train import (
    run_v6_clarity_guard_training,
)
from lykenox_voice_engine.training.speech_vocoder_v6_rejection import (
    V6_PERCEPTUALLY_REJECTED,
    V6_REJECTION_REASON,
    V6_TRAINING_ENABLED,
)
from lykenox_voice_engine.training.speech_vocoder_v6_train import (
    run_bounded_resumable_v6_training,
)


class VocoderV6PerceptualRejectionTests(unittest.TestCase):
    def test_public_model_contract_admits_hidden_sample_rate_shortcuts(self) -> None:
        self.assertFalse(LykenoxVocoderGeneratorV6.source_free)
        self.assertTrue(LykenoxVocoderGeneratorV6.sample_phase_conditioning)
        self.assertTrue(
            LykenoxVocoderGeneratorV6.deterministic_unvoiced_noise_conditioning
        )
        self.assertTrue(LykenoxVocoderGeneratorV6.local_unit_rms_shape_normalization)
        self.assertTrue(LykenoxVocoderGeneratorV6.perceptually_rejected)

    def test_all_v6_training_entrypoints_fail_before_side_effects(self) -> None:
        self.assertFalse(V6_TRAINING_ENABLED)
        self.assertTrue(V6_PERCEPTUALLY_REJECTED)
        self.assertIn("full-utterance", V6_REJECTION_REASON)
        self.assertIn("v4.2", V6_REJECTION_REASON)

        impossible_root = Path("this-path-must-never-be-read")
        with self.assertRaisesRegex(RuntimeError, "perceptually rejected"):
            run_bounded_resumable_v6_training(impossible_root)
        with self.assertRaisesRegex(RuntimeError, "perceptually rejected"):
            run_v6_clarity_guard_training(impossible_root)

    def test_retired_architecture_smoke_reports_rejection_without_data_access(self) -> None:
        result = run_v6_architecture_smoke(Path("this-path-must-never-be-read"))
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["source_free"])
        self.assertTrue(result["perceptually_rejected"])
        self.assertFalse(result["persistent_training_started"])
        self.assertFalse(result["historical_checkpoints_mutated"])


if __name__ == "__main__":
    unittest.main()
