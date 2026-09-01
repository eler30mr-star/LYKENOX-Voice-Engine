from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.training.speech_vocoder_loss_v2_weight_contract_sensitivity_audit import (
    ALIGNMENT_RETENTION_FRACTION,
    AUTHORITY_RETENTION_FRACTION,
    CANDIDATE_WEIGHTS,
    MAX_CANDIDATE_DERIVATION_RELATIVE_ERROR,
    MAX_DOMINANCE_EXPANSION,
    OBJECTIVES,
    RELATIVE_PERTURBATION,
    _evaluate_weights,
    build_weight_scenarios,
)


class VocoderLossV2WeightContractSensitivityAuditTests(unittest.TestCase):
    def test_candidate_is_human_readable_calibrated_vector(self) -> None:
        self.assertEqual(
            CANDIDATE_WEIGHTS,
            {
                "reconstruction": 1.0,
                "envelope": 3.1475,
                "presence": 19.3369,
                "spectral_balance": 60.9496,
            },
        )
        self.assertEqual(RELATIVE_PERTURBATION, 0.10)
        self.assertEqual(MAX_CANDIDATE_DERIVATION_RELATIVE_ERROR, 0.005)
        self.assertEqual(AUTHORITY_RETENTION_FRACTION, 0.75)
        self.assertEqual(ALIGNMENT_RETENTION_FRACTION, 0.75)
        self.assertEqual(MAX_DOMINANCE_EXPANSION, 1.20)

    def test_scenarios_cover_baseline_one_at_a_time_and_relative_corners(self) -> None:
        scenarios = build_weight_scenarios()
        self.assertIn("baseline", scenarios)
        self.assertGreaterEqual(len(scenarios), 20)
        for name in OBJECTIVES:
            self.assertIn(f"{name}_minus_10pct", scenarios)
            self.assertIn(f"{name}_plus_10pct", scenarios)
        for weights in scenarios.values():
            self.assertEqual(weights["reconstruction"], 1.0)
            for objective in OBJECTIVES:
                self.assertGreater(weights[objective], 0.0)

    def test_weight_evaluation_reports_positive_authority_for_compatible_gradients(self) -> None:
        gradients = {
            "reconstruction": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "envelope": torch.tensor([[0.4, 1.0, 0.0, 0.0]]),
            "presence": torch.tensor([[0.1, 0.1, 1.0, 0.0]]),
            "spectral_balance": torch.tensor([[0.05, 0.05, 0.7, 1.0]]),
        }
        result = _evaluate_weights(gradients, CANDIDATE_WEIGHTS)
        self.assertTrue(result["combined_finite_nonzero"])
        self.assertAlmostEqual(sum(result["shares"].values()), 1.0, places=6)
        for name in OBJECTIVES:
            self.assertGreater(result["shares"][name], 0.0)
            self.assertGreater(result["alignments"][name], 0.0)
            self.assertGreater(result["descent_dots"][name], 0.0)


if __name__ == "__main__":
    unittest.main()
