from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import (
    speech_vocoder_minimum_phase_directional_weight_calibration as calibration,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_heldout_audio as heldout
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_and_listen as pipeline
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_v3 as trainer
from lykenox_voice_engine.training import (
    speech_vocoder_minimum_phase_train_and_listen_contract_v2 as contract,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_objective_v3 import (
    OwnedMinimumPhaseObjectiveV3,
)


class DirectionalMinimumPhaseV3Tests(unittest.TestCase):
    def test_identity_gram_selects_balanced_positive_static_weights(self) -> None:
        probe = calibration.GradientGramProbe(
            "identity", "parameter_space", "neutral", torch.eye(4, dtype=torch.float64)
        )
        weights, report = calibration.calibrate_directional_fixed_weights([probe])
        self.assertIsNotNone(weights)
        self.assertEqual(report["status"], "pass")
        assert weights is not None
        values = weights.as_dict()
        self.assertEqual(values["reconstruction"], 1.0)
        self.assertGreater(min(values.values()), 0.0)
        summary = calibration.summarize_weights(weights, [probe])
        self.assertGreater(summary["worst_alignment"], 0.0)
        shares = summary["mean_weighted_gradient_norm_shares"]
        self.assertLess(max(shares.values()) - min(shares.values()), 1.0e-12)

    def test_opposite_objectives_refuse_fake_static_solution(self) -> None:
        gram = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        probe = calibration.GradientGramProbe("conflict", "cepstrum_space", "neutral", gram)
        weights, report = calibration.calibrate_directional_fixed_weights([probe])
        self.assertIsNone(weights)
        self.assertEqual(report["status"], "no_static_positive_weight_solution")

    def test_contract_allows_only_pretrain_fixed_calibration(self) -> None:
        self.assertEqual(contract.POLICY_ID, "LYX-POL-001")
        self.assertTrue(contract.CPU_ONLY)
        self.assertTrue(contract.DIRECTIONAL_PRETRAIN_CALIBRATION_REQUIRED)
        self.assertTrue(contract.CALIBRATION_BEFORE_MODEL_OPTIMIZER_REQUIRED)
        self.assertTrue(contract.CALIBRATED_WEIGHTS_FIXED_FOR_RUN)
        self.assertFalse(contract.ADAPTIVE_REWEIGHTING_DURING_TRAINING_AUTHORIZED)
        self.assertFalse(contract.RUNTIME_WEIGHT_REDERIVATION_AFTER_TRAINING_START_AUTHORIZED)
        self.assertFalse(contract.THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED)
        self.assertFalse(contract.POSTHOC_GAIN_NORMALIZATION_AUTHORIZED)
        self.assertFalse(contract.POSTHOC_EQ_AUTHORIZED)
        self.assertFalse(contract.POSTHOC_DENOISING_AUTHORIZED)
        self.assertFalse(contract.METRICS_CAN_ACCEPT_VOICE_QUALITY)

    def test_v3_objective_uses_supplied_fixed_weights_only(self) -> None:
        source = inspect.getsource(OwnedMinimumPhaseObjectiveV3)
        self.assertIn("self.weights.reconstruction", source)
        self.assertIn("self.weights.envelope", source)
        self.assertIn("self.weights.presence", source)
        self.assertIn("self.weights.spectral_balance", source)
        self.assertNotIn("FROZEN_MINIMUM_PHASE_WEIGHTS", source)
        self.assertNotIn("combine_owned_minimum_phase_loss_v2", source)

    def test_trainer_calibrates_and_verifies_before_fresh_optimizer(self) -> None:
        source = inspect.getsource(trainer.run_minimum_phase_training_v3)
        fresh = source.split("else:", 1)[1]
        calibration_index = fresh.index("run_directional_calibration")
        optimizer_index = fresh.index("optimizer = torch.optim.AdamW")
        self.assertLess(calibration_index, optimizer_index)
        self.assertIn('"device": "cpu"', inspect.getsource(trainer._run_config))
        self.assertIn("fixed_weights_from_mapping", inspect.getsource(trainer))
        lowered = inspect.getsource(trainer).lower()
        for forbidden in ("from_pretrained", "vocos", "bigvgan", "hifigan"):
            self.assertNotIn(forbidden, lowered)

    def test_one_shot_and_heldout_are_v3_best_only(self) -> None:
        self.assertEqual(
            pipeline.PIPELINE_VERSION,
            "owned-minimum-phase-train-and-listen-v3-directional-fixed",
        )
        pipeline_source = inspect.getsource(pipeline)
        self.assertIn("run_minimum_phase_training_v3", pipeline_source)
        heldout_source = inspect.getsource(heldout)
        self.assertIn('checkpoint.name != "best.pt"', heldout_source)
        self.assertIn("fixed_weights_from_mapping", heldout_source)
        self.assertIn("OwnedMinimumPhaseObjectiveV3", heldout_source)
        self.assertIn('subtype="FLOAT"', heldout_source)
        lowered = heldout_source.lower()
        self.assertNotIn("normalize(", lowered)
        self.assertNotIn("equalizer", lowered)
        self.assertNotIn("from_pretrained", lowered)


if __name__ == "__main__":
    unittest.main()
