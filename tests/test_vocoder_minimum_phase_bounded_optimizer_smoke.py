from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_architecture_contract as contract
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_bounded_optimizer_smoke as smoke


class VocoderMinimumPhaseBoundedOptimizerSmokeTests(unittest.TestCase):
    def test_historical_scope_was_exactly_two_updates_on_one_owned_segment(self) -> None:
        self.assertTrue(smoke.BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED)
        self.assertEqual(smoke.SEGMENT_MEL_FRAMES, 32)
        self.assertEqual(smoke.MAX_ITEMS, 1)
        self.assertEqual(smoke.MAX_UPDATES, 2)
        self.assertEqual(smoke.SEGMENT_MEL_FRAMES, contract.BOUNDED_OPTIMIZER_SMOKE_SEGMENT_FRAMES)
        self.assertEqual(smoke.MAX_ITEMS, contract.BOUNDED_OPTIMIZER_SMOKE_MAX_ITEMS)
        self.assertEqual(smoke.MAX_UPDATES, contract.BOUNDED_OPTIMIZER_SMOKE_MAX_UPDATES)
        self.assertGreater(smoke.LEARNING_RATE, 0.0)
        self.assertLessEqual(smoke.LEARNING_RATE, 2.0e-4)
        self.assertEqual(smoke.MAX_GRAD_NORM, 1.0)

    def test_current_contract_records_pass_and_closes_repeat_authorization(self) -> None:
        self.assertEqual(contract.BOUNDED_OPTIMIZER_SMOKE_STATUS, "pass")
        self.assertTrue(contract.BOUNDED_OPTIMIZER_SMOKE_CONSUMED)
        self.assertFalse(contract.BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED)
        self.assertTrue(contract.PARAMETER_SPACE_GRADIENT_AUDIT_AUTHORIZED)
        self.assertFalse(contract.EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED)

    def test_persistent_training_checkpoint_and_trainer_remain_forbidden(self) -> None:
        self.assertFalse(smoke.TRAINER_IMPLEMENTATION_AUTHORIZED)
        self.assertFalse(smoke.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(smoke.NEW_VOCODER_CHECKPOINT_AUTHORIZED)
        self.assertFalse(contract.OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(contract.TRAINER_IMPLEMENTATION_AUTHORIZED)
        self.assertFalse(contract.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(contract.NEW_VOCODER_CHECKPOINT_AUTHORIZED)

    def test_smoke_uses_owned_v2_data_and_frozen_four_objective_loss(self) -> None:
        source = inspect.getsource(smoke)
        self.assertIn("collect_owned_vocoder_segments", source)
        self.assertIn("valid_context_multi_resolution_reconstruction_loss", source)
        self.assertIn("ConditioningAlignedLogMelEnvelopeLossV2", source)
        self.assertIn("target_relative_presence_loss_v2", source)
        self.assertIn("target_relative_spectral_balance_loss", source)
        self.assertIn("combine_owned_vocoder_loss_v2", source)
        self.assertIn("FROZEN_WEIGHTS", source)
        self.assertIn("render_owned_minimum_phase_vocoder_path", source)

    def test_smoke_has_only_ephemeral_optimizer_and_no_checkpoint_io(self) -> None:
        source = inspect.getsource(smoke).lower()
        self.assertIn("torch.optim.sgd", source)
        self.assertIn("clip_grad_norm_", source)
        self.assertIn("optimizer.step()", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("torch.load(", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("vocos", source)
        self.assertNotIn("bigvgan", source)
        self.assertNotIn("hifigan", source)
        self.assertNotIn("atomic_json", source)

    def test_smoke_reports_rejection_gates_not_quality_acceptance(self) -> None:
        source = inspect.getsource(smoke)
        for required in (
            '"total_loss_decreased"',
            '"no_severe_grid_excess_before"',
            '"no_severe_grid_excess_after"',
            '"checkpoints_unchanged"',
            '"persistent_training_started": False',
            '"persistent_training_authorized": False',
            '"new_vocoder_checkpoint_authorized": False',
            '"metrics_accept_voice_quality": False',
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
