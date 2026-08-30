from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_artifact import (
    ISOLATED_MEL_ARTIFACT_VERSION,
    TRAINABLE_PREFIX,
    freeze_except_mel_decoder,
    require_frozen_state_exact,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_resume_smoke import (
    SMOKE_VERSION,
    run_isolated_mel_resume_smoke,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_train import (
    HARD_EPOCH_LIMIT,
    TRAINER_CONTRACT_VERSION,
    _run_config,
    run_isolated_mel_fidelity_training,
)


class AcousticMelFidelityTrainingContractTests(unittest.TestCase):
    def test_only_mel_decoder_can_be_trainable(self) -> None:
        model = LykenoxSpeechAcousticModel(LykenoxSpeechConfig(vocab_size=16))
        names = freeze_except_mel_decoder(model)
        self.assertTrue(names)
        self.assertTrue(all(name.startswith(TRAINABLE_PREFIX) for name in names))
        self.assertTrue(all(
            parameter.requires_grad == name.startswith(TRAINABLE_PREFIX)
            for name, parameter in model.named_parameters()
        ))

    def test_frozen_state_guard_detects_non_mel_mutation(self) -> None:
        base = LykenoxSpeechAcousticModel(LykenoxSpeechConfig(vocab_size=16))
        candidate = LykenoxSpeechAcousticModel(LykenoxSpeechConfig(vocab_size=16))
        candidate.load_state_dict(base.state_dict())
        with torch.no_grad():
            candidate.embedding.weight[1, 0] += 1.0
        with self.assertRaises(RuntimeError):
            require_frozen_state_exact(candidate, base)

    def test_trainer_identity_is_one_epoch_and_isolated(self) -> None:
        self.assertEqual(
            TRAINER_CONTRACT_VERSION,
            "acoustic-mel-fidelity-first-epoch-resumable-v1",
        )
        self.assertEqual(ISOLATED_MEL_ARTIFACT_VERSION, "acoustic-mel-fidelity-isolated-artifact-v1")
        self.assertEqual(HARD_EPOCH_LIMIT, 1)
        config = _run_config(
            base_sha256="abc",
            train_count=10,
            val_count=2,
            batch_size=2,
            seed=1337,
            learning_rate=1e-4,
            grad_clip=5.0,
            checkpoint_every_updates=2,
            dataset_item_limit=None,
        )
        self.assertEqual(config["trainable_parameter_prefix"], TRAINABLE_PREFIX)
        self.assertEqual(config["hard_epoch_limit"], 1)
        self.assertTrue(config["teacher_duration_grid"])
        self.assertFalse(config["duration_training"])
        self.assertFalse(config["prosody_training"])
        self.assertFalse(config["vocoder_training"])

    def test_trainer_and_resume_smoke_keep_epoch2_closed(self) -> None:
        trainer_source = inspect.getsource(run_isolated_mel_fidelity_training)
        self.assertIn('"epoch2_training_blocked": True', trainer_source)
        self.assertIn('if epoch > HARD_EPOCH_LIMIT or history:', trainer_source)
        self.assertIn("require_frozen_state_exact", trainer_source)
        self.assertNotIn("model.parameters(), lr=learning_rate", trainer_source)

        self.assertEqual(SMOKE_VERSION, "acoustic-mel-fidelity-exact-resume-smoke-v1")
        smoke_source = inspect.getsource(run_isolated_mel_resume_smoke)
        self.assertIn("TemporaryDirectory", smoke_source)
        self.assertIn('"persistent_training_not_started"', smoke_source)
        self.assertIn('"training_authorized": False', smoke_source)
        self.assertIn("gate_checkpoint_unchanged_on_rerun", smoke_source)


if __name__ == "__main__":
    unittest.main()
