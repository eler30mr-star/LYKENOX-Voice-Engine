from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_objective as objective


class MinimumPhaseObjectiveTests(unittest.TestCase):
    def test_active_objective_is_v2_and_v1_is_forbidden(self) -> None:
        self.assertEqual(objective.ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION, "owned-minimum-phase-objective-v2")
        self.assertEqual(
            objective.ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
            "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2",
        )
        self.assertFalse(objective.HISTORICAL_WAVEFORM_WEIGHT_CONTRACT_AUTHORIZED)
        self.assertFalse(objective.ADAPTIVE_REWEIGHTING_AUTHORIZED)
        self.assertFalse(objective.RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED)
        self.assertEqual(
            objective.active_weights(),
            {
                "reconstruction": 1.0,
                "envelope": 16.1348,
                "presence": 4.0842,
                "spectral_balance": 3.3202,
            },
        )

    def test_source_contains_no_v1_weight_combiner(self) -> None:
        source = inspect.getsource(objective)
        self.assertNotIn("combine_owned_vocoder_loss_v2", source)
        self.assertNotIn("FROZEN_WEIGHTS", source)
        self.assertIn("combine_owned_minimum_phase_loss_v2", source)
        self.assertIn("FROZEN_MINIMUM_PHASE_WEIGHTS", source)

    def test_result_contract_is_scalar(self) -> None:
        result = objective.MinimumPhaseObjectiveResult(
            total=torch.tensor(1.0),
            reconstruction=torch.tensor(2.0),
            envelope=torch.tensor(3.0),
            presence=torch.tensor(4.0),
            spectral_balance=torch.tensor(5.0),
        )
        self.assertEqual(
            result.detached_terms(),
            {
                "reconstruction": 2.0,
                "envelope": 3.0,
                "presence": 4.0,
                "spectral_balance": 5.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
