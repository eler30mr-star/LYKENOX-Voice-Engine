from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.models.speech.mel_detail_head import MEL_DETAIL_HEAD_ARCHITECTURE_V1
from lykenox_voice_engine.training.speech_acoustic_mel_detail_head_artifact import (
    MEL_DETAIL_HEAD_CHECKPOINT_KIND,
    MEL_DETAIL_HEAD_CHECKPOINT_VERSION,
    save_mel_detail_head_checkpoint,
)
from lykenox_voice_engine.training.speech_acoustic_mel_detail_head_resume_smoke import (
    SMOKE_VERSION,
    run_mel_detail_head_resume_smoke,
)
from lykenox_voice_engine.training.speech_acoustic_mel_detail_head_train import (
    HARD_EPOCH_LIMIT,
    TRAINER_CONTRACT_VERSION,
    _run_config,
    run_mel_detail_head_training,
)


class AcousticMelDetailHeadTrainingContractTests(unittest.TestCase):
    def test_run_identity_is_detail_head_only_and_one_epoch(self) -> None:
        self.assertEqual(
            TRAINER_CONTRACT_VERSION,
            "acoustic-frame-hidden-mel-detail-first-epoch-resumable-v1",
        )
        self.assertEqual(MEL_DETAIL_HEAD_CHECKPOINT_VERSION, 1)
        self.assertEqual(
            MEL_DETAIL_HEAD_CHECKPOINT_KIND,
            "lykenox_acoustic_frame_hidden_mel_detail_first_epoch_checkpoint",
        )
        self.assertEqual(HARD_EPOCH_LIMIT, 1)
        config = _run_config(
            base_sha256="abc",
            train_count=118,
            val_count=14,
            batch_size=2,
            seed=1901,
            learning_rate=1e-4,
            grad_clip=5.0,
            checkpoint_every_updates=8,
            dataset_item_limit=None,
        )
        self.assertEqual(config["architecture"], MEL_DETAIL_HEAD_ARCHITECTURE_V1)
        self.assertEqual(config["base_checkpoint_sha256"], "abc")
        self.assertEqual(config["trainable_surface"], "detail_head_only")
        self.assertEqual(config["hard_epoch_limit"], 1)
        self.assertTrue(config["teacher_duration_grid"])
        self.assertFalse(config["duration_training"])
        self.assertFalse(config["prosody_training"])
        self.assertFalse(config["vocoder_training"])

    def test_artifact_persists_detail_head_not_acoustic_model(self) -> None:
        source = inspect.getsource(save_mel_detail_head_checkpoint)
        self.assertIn('"detail_head_state": candidate.detail_head.state_dict()', source)
        self.assertIn('"optimizer_state": optimizer.state_dict()', source)
        self.assertIn('"torch_rng_state": torch.get_rng_state()', source)
        self.assertNotIn('"model_state"', source)
        self.assertNotIn('"base_model_state"', source)
        self.assertIn("candidate.base_model.parameters()", source)

    def test_trainer_keeps_base_immutable_and_epoch_two_closed(self) -> None:
        source = inspect.getsource(run_mel_detail_head_training)
        self.assertIn("candidate.detail_head.parameters()", source)
        self.assertNotIn("candidate.parameters(), lr=", source)
        self.assertIn("candidate.base_model.parameters()", source)
        self.assertIn("if epoch > HARD_EPOCH_LIMIT or history:", source)
        self.assertIn('"epoch2_training_blocked": True', source)
        self.assertIn('"duration_training": False', source)
        self.assertIn('"prosody_training": False', source)
        self.assertIn('"vocoder_training": False', source)
        self.assertIn('"run_frame_hidden_detail_full_utterance_v4_2_ab"', source)

    def test_resume_smoke_is_temporary_exact_and_non_authorizing(self) -> None:
        self.assertEqual(
            SMOKE_VERSION,
            "acoustic-frame-hidden-mel-detail-exact-resume-smoke-v1",
        )
        source = inspect.getsource(run_mel_detail_head_resume_smoke)
        self.assertIn("TemporaryDirectory", source)
        self.assertIn("detail_head_state_exact", source)
        self.assertIn("optimizer_state_exact", source)
        self.assertIn("torch_rng_state_exact", source)
        self.assertIn("metadata_exact", source)
        self.assertIn("base_identity_exact", source)
        self.assertIn("gate_checkpoint_unchanged_on_rerun", source)
        self.assertIn('"persistent_training_not_started"', source)
        self.assertIn('"training_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
