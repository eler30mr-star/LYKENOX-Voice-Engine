from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_loss_v2_weight_contract as contract


class VocoderLossV2WeightContractTests(unittest.TestCase):
    def test_exact_frozen_weights_and_versions(self) -> None:
        self.assertEqual(
            contract.OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION,
            "owned-vocoder-loss-v2-weight-contract-v1",
        )
        self.assertEqual(
            contract.DATA_CONTRACT_VERSION,
            "vocoder-segment-v2-full-utterance-mel-pitch-conditioning",
        )
        self.assertEqual(
            contract.LOSS_CONTRACT_VERSION,
            "owned-vocoder-loss-v2-valid-context-conditioning-aligned",
        )
        self.assertEqual(
            contract.PRESENCE_CONTRACT_VERSION,
            "owned-vocoder-presence-v2-valid-context-target-relative",
        )
        self.assertEqual(
            contract.FROZEN_WEIGHTS.as_dict(),
            {
                "reconstruction": 1.0,
                "envelope": 3.1475,
                "presence": 19.3369,
                "spectral_balance": 60.9496,
            },
        )

    def test_sensitivity_evidence_supports_freeze(self) -> None:
        self.assertEqual(contract.SENSITIVITY_AUDIT_STATUS, "pass")
        self.assertEqual(contract.SENSITIVITY_SCENARIO_COUNT, 23)
        self.assertEqual(contract.SENSITIVITY_RELATIVE_WEIGHT_PERTURBATION, 0.10)
        self.assertTrue(all(contract.SENSITIVITY_GATES.values()))
        for value in contract.CANDIDATE_DERIVATION_RELATIVE_ERRORS.values():
            self.assertLess(value, 0.005)
        for value in contract.SENSITIVITY_METRICS[
            "all_scenarios_minimum_weighted_gradient_norm_shares"
        ].values():
            self.assertGreater(value, 0.08)
        for value in contract.SENSITIVITY_METRICS[
            "all_scenarios_minimum_combined_gradient_alignment_cosines"
        ].values():
            self.assertGreater(value, 0.0)
        for value in contract.SENSITIVITY_METRICS[
            "all_scenarios_minimum_first_order_descent_dots"
        ].values():
            self.assertGreater(value, 0.0)
        self.assertLess(
            contract.SENSITIVITY_METRICS[
                "all_scenarios_maximum_weighted_gradient_norm_share"
            ],
            0.60,
        )

    def test_combiner_uses_exact_contract(self) -> None:
        reconstruction = torch.tensor(1.0)
        envelope = torch.tensor(2.0)
        presence = torch.tensor(3.0)
        spectral_balance = torch.tensor(4.0)
        expected = (
            1.0
            + 3.1475 * 2.0
            + 19.3369 * 3.0
            + 60.9496 * 4.0
        )
        actual = contract.combine_owned_vocoder_loss_v2(
            reconstruction=reconstruction,
            envelope=envelope,
            presence=presence,
            spectral_balance=spectral_balance,
        )
        self.assertAlmostEqual(float(actual), expected, places=5)

    def test_contract_rejects_silent_dynamic_reweighting_and_model_work(self) -> None:
        self.assertFalse(contract.ADAPTIVE_LOSS_REWEIGHTING_AUTHORIZED)
        self.assertFalse(contract.RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED)
        self.assertFalse(
            contract.AUTOMATIC_WEIGHT_REDERIVATION_DURING_TRAINING_AUTHORIZED
        )
        self.assertFalse(contract.MODEL_INSTANTIATION_AUTHORIZED)
        self.assertFalse(contract.OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(contract.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(contract.NEW_VOCODER_ARCHITECTURE_AUTHORIZED)
        self.assertFalse(contract.THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED)

    def test_contract_contains_no_training_or_external_model_path(self) -> None:
        source = inspect.getsource(contract).lower()
        for forbidden in (
            "torch.optim.",
            ".backward(",
            ".step(",
            "torch.save(",
            "from_pretrained",
            "vocos",
            "bigvgan",
            "hifigan",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
