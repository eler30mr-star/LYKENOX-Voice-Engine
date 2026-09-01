from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_heldout_audio as heldout
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_noise as noise
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_v2 as trainer
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_and_listen as pipeline
from lykenox_voice_engine.training import (
    speech_vocoder_minimum_phase_train_and_listen_contract as contract,
)


class MinimumPhaseTrainAndListenV2Tests(unittest.TestCase):
    def test_scoped_policy_contract_is_cpu_bounded_and_not_general_persistent_training(self) -> None:
        self.assertEqual(contract.POLICY_ID, "LYX-POL-001")
        self.assertTrue(contract.CPU_ONLY)
        self.assertTrue(contract.TRAIN_AND_LISTEN_AUTHORIZED)
        self.assertEqual(contract.MAX_UPDATES_AUTHORIZED, 400)
        self.assertTrue(contract.V2_AUTHORITY_PREFLIGHT_REQUIRED)
        self.assertTrue(contract.EXACT_RESUME_REQUIRED)
        self.assertTrue(contract.SCOPED_CHECKPOINT_CREATION_AUTHORIZED)
        self.assertFalse(contract.GENERAL_PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(contract.GENERAL_OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(contract.THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED)
        self.assertFalse(contract.POSTHOC_GAIN_NORMALIZATION_AUTHORIZED)
        self.assertFalse(contract.POSTHOC_EQ_AUTHORIZED)
        self.assertFalse(contract.POSTHOC_DENOISING_AUTHORIZED)
        self.assertFalse(contract.METRICS_CAN_ACCEPT_VOICE_QUALITY)

    def test_per_example_noise_seed_is_stable_but_not_reused_for_other_crops(self) -> None:
        first = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0001", start_frame=12
        )
        repeat = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0001", start_frame=12
        )
        other_utterance = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0002", start_frame=12
        )
        other_crop = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0001", start_frame=13
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other_utterance)
        self.assertNotEqual(first, other_crop)

    def test_trainer_v2_runs_authority_preflight_before_optimizer_creation(self) -> None:
        source = inspect.getsource(trainer)
        self.assertIn("run_v2_authority_preflight", source)
        self.assertIn("OwnedMinimumPhaseObjectiveV2", source)
        self.assertIn("stable_owned_noise_seed", source)
        self.assertIn('"device": "cpu"', source)
        preflight_index = source.index("preflight = run_v2_authority_preflight")
        optimizer_index = source.index("optimizer = torch.optim.AdamW", preflight_index)
        self.assertLess(preflight_index, optimizer_index)
        self.assertNotIn("speech_vocoder_loss_v2_weight_contract import", source)
        self.assertNotIn("combine_owned_vocoder_loss_v2", source)
        lowered = source.lower()
        self.assertNotIn("from_pretrained", lowered)
        self.assertNotIn("vocos", lowered)
        self.assertNotIn("bigvgan", lowered)
        self.assertNotIn("hifigan", lowered)

    def test_one_shot_pipeline_renders_only_after_training_pass(self) -> None:
        source = inspect.getsource(pipeline)
        self.assertIn("run_minimum_phase_training_v2", source)
        self.assertIn('if training.get("status") != "pass"', source)
        self.assertIn('Path(best_checkpoint)', source)
        self.assertEqual(pipeline.PIPELINE_VERSION, "owned-minimum-phase-train-and-listen-v2")

    def test_heldout_requires_best_and_uses_per_utterance_noise_without_posthoc(self) -> None:
        source = inspect.getsource(heldout)
        lowered = source.lower()
        self.assertIn('checkpoint.name != "best.pt"', source)
        self.assertIn("stable_owned_noise_seed", source)
        self.assertIn('subtype="FLOAT"', source)
        self.assertIn('"product_acceptance_requires_human_listening": True', source)
        self.assertNotIn("last_fallback_no_best", source)
        self.assertNotIn("normalize(", lowered)
        self.assertNotIn("equalizer", lowered)
        self.assertNotIn("from_pretrained", lowered)


if __name__ == "__main__":
    unittest.main()
