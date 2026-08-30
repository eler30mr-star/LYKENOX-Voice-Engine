from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_smoke import (
    LEARNING_RATE,
    SMOKE_VERSION,
    UPDATES,
    run_acoustic_mel_fidelity_smoke,
)


class AcousticMelFidelitySmokeContractTests(unittest.TestCase):
    def test_smoke_identity_is_bounded(self) -> None:
        self.assertEqual(SMOKE_VERSION, "acoustic-mel-fidelity-isolated-smoke-v1")
        self.assertEqual(UPDATES, 12)
        self.assertGreater(LEARNING_RATE, 0.0)

    def test_smoke_only_trains_mel_decoder_and_is_nonpersistent(self) -> None:
        source = inspect.getsource(run_acoustic_mel_fidelity_smoke)
        self.assertIn("model.mel_decoder.parameters()", source)
        self.assertIn('name.startswith("mel_decoder.")', source)
        self.assertIn('"duration_prediction_exact"', source)
        self.assertIn('"f0_prediction_exact"', source)
        self.assertIn('"voicing_logits_exact"', source)
        self.assertIn('"protected_checkpoints_unchanged"', source)
        self.assertIn('"persistent_training_started": False', source)
        self.assertIn('"training_authorized": False', source)
        self.assertIn('"vocoder_modified": False', source)
        self.assertIn('"predicted_duration_modified": False', source)
        for forbidden in (
            "torch.save(",
            "save_acoustic_prosody_checkpoint",
            "optimizer = torch.optim.AdamW(model.parameters()",
            '"training_authorized": True',
            '"persistent_training_started": True',
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
