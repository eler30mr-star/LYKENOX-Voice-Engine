from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import (
    speech_vocoder_minimum_phase_weight_recalibration_audit as audit,
)


class VocoderMinimumPhaseWeightRecalibrationAuditTests(unittest.TestCase):
    def test_contract_preserves_v1_and_only_proposes_v2_candidate(self) -> None:
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-minimum-phase-architecture-weight-recalibration-audit-v1",
        )
        self.assertEqual(
            audit.CANDIDATE_CONTRACT_VERSION,
            "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2-candidate",
        )
        self.assertEqual(audit.DERIVATION_SPACE, "cepstrum_space")
        self.assertEqual(audit.CROSS_CHECK_SPACE, "parameter_space")
        self.assertEqual(
            audit.WAVEFORM_SPACE_V1_VERSION,
            "owned-vocoder-loss-v2-weight-contract-v1",
        )
        self.assertFalse(audit.WEIGHT_CONTRACT_V2_AUTHORIZED)
        self.assertFalse(audit.EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED)
        self.assertFalse(audit.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(audit.NEW_VOCODER_CHECKPOINT_AUTHORIZED)

    def test_derivation_equalizes_gradient_norm_scale_without_hand_tuned_shares(self) -> None:
        weights = audit._derive_equalized_weights(
            {
                "reconstruction": 12.0,
                "envelope": 3.0,
                "presence": 6.0,
                "spectral_balance": 2.0,
            }
        )
        self.assertEqual(
            weights,
            {
                "reconstruction": 1.0,
                "envelope": 4.0,
                "presence": 2.0,
                "spectral_balance": 6.0,
            },
        )

    def test_audit_reuses_owned_real_data_and_both_jacobian_spaces(self) -> None:
        source = inspect.getsource(audit)
        self.assertIn("collect_owned_vocoder_segments", source)
        self.assertIn('DERIVATION_SPACE = "cepstrum_space"', source)
        self.assertIn('CROSS_CHECK_SPACE = "parameter_space"', source)
        self.assertIn("cepstrum_gradients", source)
        self.assertIn("parameter_gradients", source)
        self.assertIn("derived_cepstrum_space_candidate_weights", source)
        self.assertIn("derived_parameter_space_cross_check_weights", source)
        self.assertIn("candidate_in_cepstrum_space", source)
        self.assertIn("candidate_in_parameter_space", source)

    def test_audit_has_no_optimizer_update_or_checkpoint_route(self) -> None:
        source = inspect.getsource(audit).lower()
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
        self.assertIn('"optimizer_created": false', source)
        self.assertIn('"parameter_update_executed": false', source)
        self.assertIn('"weight_contract_v2_authorized": false', source)


if __name__ == "__main__":
    unittest.main()
