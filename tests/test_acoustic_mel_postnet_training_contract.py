from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.models.speech.mel_postnet import MEL_POSTNET_ARCHITECTURE_V1
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_artifact import (
    MEL_POSTNET_CHECKPOINT_KIND,
    MEL_POSTNET_CHECKPOINT_VERSION,
    MEL_POSTNET_HIDDEN_CHANNELS,
    save_mel_postnet_checkpoint,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_resume_smoke import (
    SMOKE_VERSION,
    run_mel_postnet_resume_smoke,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_train import (
    HARD_EPOCH_LIMIT,
    TRAINER_CONTRACT_VERSION,
    _run_config,
    run_mel_postnet_training,
)


class AcousticMelPostnetTrainingContractTests(unittest.TestCase):
    def test_run_identity_is_postnet_only_and_one_epoch(self) -> None:
        self.assertEqual(
            TRAINER_CONTRACT_VERSION,
            "acoustic-mel-postnet-first-epoch-resumable-v1",
        )
        self.assertEqual(MEL_POSTNET_CHECKPOINT_VERSION, 1)
        self.assertEqual(
            MEL_POSTNET_CHECKPOINT_KIND,
            "lykenox_acoustic_mel_postnet_first_epoch_checkpoint",
        )
        self.assertEqual(MEL_POSTNET_HIDDEN_CHANNELS, 128)
        self.assertEqual(HARD_EPOCH_LIMIT, 1)
        config = _run_config(
            base_sha256="abc",
            train_count=118,
            val_count=14,
            batch_size=2,
            seed=1701,
            learning_rate=1e-4,
            grad_clip=5.0,
            checkpoint_every_updates=8,
            dataset_item_limit=None,
        )
        self.assertEqual(config["architecture"], MEL_POSTNET_ARCHITECTURE_V1)
        self.assertEqual(config["base_checkpoint_sha256"], "abc")
        self.assertEqual(config["trainable_surface"], "postnet_only")
        self.assertEqual(config["hard_epoch_limit"], 1)
        self.assertTrue(config["teacher_duration_grid"])
        self.assertFalse(config["duration_training"])
        self.assertFalse(config["prosody_training"])
        self.assertFalse(config["vocoder_training"])

    def test_artifact_persists_postnet_not_trainable_acoustic_copy(self) -> None:
        source = inspect.getsource(save_mel_postnet_checkpoint)
        self.assertIn('"postnet_state": candidate.postnet.state_dict()', source)
        self.assertIn('"optimizer_state": optimizer.state_dict()', source)
        self.assertIn('"torch_rng_state": torch.get_rng_state()', source)
        self.assertNotIn('"model_state"', source)
        self.assertIn("candidate.base_model.parameters()", source)

    def test_trainer_keeps_base_immutable_and_epoch_two_closed(self) -> None:
        source = inspect.getsource(run_mel_postnet_training)
        self.assertIn("candidate.postnet.parameters()", source)
        self.assertNotIn("candidate.parameters(), lr=", source)
        self.assertIn("candidate.base_model.parameters()", source)
        self.assertIn("if epoch > HARD_EPOCH_LIMIT or history:", source)
        self.assertIn('"epoch2_training_blocked": True', source)
        self.assertIn('"duration_training": False', source)
        self.assertIn('"prosody_training": False', source)
        self.assertIn('"vocoder_training": False', source)

    def test_resume_smoke_is_temporary_and_non_authorizing(self) -> None:
        self.assertEqual(SMOKE_VERSION, "acoustic-mel-postnet-exact-resume-smoke-v1")
        source = inspect.getsource(run_mel_postnet_resume_smoke)
        self.assertIn("TemporaryDirectory", source)
        self.assertIn("postnet_state_exact", source)
        self.assertIn("gate_checkpoint_unchanged_on_rerun", source)
        self.assertIn('"persistent_training_not_started"', source)
        self.assertIn('"training_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
