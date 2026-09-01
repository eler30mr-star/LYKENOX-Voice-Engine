from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import (
    speech_vocoder_loss_v2_minimum_phase_weight_contract as contract,
)


class MinimumPhaseLossWeightContractTests(unittest.TestCase):
    def test_v2_contract_is_architecture_coupled_and_fixed(self) -> None:
        self.assertEqual(
            contract.OWNED_MINIMUM_PHASE_LOSS_WEIGHT_CONTRACT_VERSION,
            "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2",
        )
        self.assertEqual(
            contract.ARCHITECTURE_FAMILY,
            "owned_minimum_phase_time_varying_filter_over_neutral_excitation",
        )
        self.assertEqual(contract.DERIVATION_SPACE, "cepstrum_space")
        self.assertEqual(contract.CROSS_CHECK_SPACE, "parameter_space")
        self.assertTrue(contract.MINIMUM_PHASE_WEIGHT_V2_IMPLEMENTED)
        self.assertTrue(contract.MINIMUM_PHASE_WEIGHT_V2_IS_ACTIVE_CANDIDATE)
        self.assertFalse(contract.ADAPTIVE_LOSS_REWEIGHTING_AUTHORIZED)
        self.assertFalse(contract.RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED)
        self.assertFalse(contract.AUTOMATIC_WEIGHT_REDERIVATION_DURING_TRAINING_AUTHORIZED)
        self.assertFalse(contract.PERSISTENT_TRAINING_AUTHORIZED)

    def test_v2_corrects_measured_authority_collapse(self) -> None:
        self.assertEqual(
            contract.FROZEN_MINIMUM_PHASE_WEIGHTS.as_dict(),
            {
                "reconstruction": 1.0,
                "envelope": 16.1348,
                "presence": 4.0842,
                "spectral_balance": 3.3202,
            },
        )
        cep = contract.PROJECTED_CEPSTRUM_MEAN_WEIGHTED_SHARES
        for value in cep.values():
            self.assertAlmostEqual(value, 0.25, places=12)
        param = contract.PROJECTED_PARAMETER_MEAN_WEIGHTED_SHARES
        self.assertGreater(min(param.values()), 0.20)
        self.assertLess(max(param.values()), 0.28)
        self.assertLess(
            contract.PARAMETER_VS_CEPSTRUM_RELATIVE_DIFFERENCE["envelope"], 0.05
        )
        self.assertLess(
            contract.PARAMETER_VS_CEPSTRUM_RELATIVE_DIFFERENCE["presence"], 0.06
        )

    def test_combiner_uses_only_v2_weights(self) -> None:
        values = {
            "reconstruction": torch.tensor(2.0, dtype=torch.float64),
            "envelope": torch.tensor(3.0, dtype=torch.float64),
            "presence": torch.tensor(5.0, dtype=torch.float64),
            "spectral_balance": torch.tensor(7.0, dtype=torch.float64),
        }
        total = contract.combine_owned_minimum_phase_loss_v2(**values)
        w = contract.FROZEN_MINIMUM_PHASE_WEIGHTS
        expected = (
            w.reconstruction * 2.0
            + w.envelope * 3.0
            + w.presence * 5.0
            + w.spectral_balance * 7.0
        )
        self.assertAlmostEqual(float(total), expected, places=12)

    def test_contract_has_no_training_or_external_model_path(self) -> None:
        source = inspect.getsource(contract).lower()
        for forbidden in (
            "torch.optim",
            "optimizer.step",
            ".backward(",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "vocos",
            "bigvgan",
            "hifigan",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
