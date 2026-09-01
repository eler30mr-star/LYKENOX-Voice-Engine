from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_loss_v2_weight_calibration_audit as audit


class OwnedVocoderLossV2WeightCalibrationAuditTests(unittest.TestCase):
    def test_equalization_weights_are_derived_from_gradient_norms(self) -> None:
        weights = audit._derive_equalized_weights(
            {
                "reconstruction": 12.0,
                "envelope": 3.0,
                "presence": 1.5,
                "spectral_balance": 0.25,
            }
        )
        self.assertEqual(weights["reconstruction"], 1.0)
        self.assertEqual(weights["envelope"], 4.0)
        self.assertEqual(weights["presence"], 8.0)
        self.assertEqual(weights["spectral_balance"], 48.0)

    def test_audit_includes_four_owned_objectives_without_authorizing_weights(self) -> None:
        source = inspect.getsource(audit)
        run_source = inspect.getsource(audit.run_owned_vocoder_loss_v2_weight_calibration_audit)
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-vocoder-loss-v2-four-objective-weight-calibration-audit-v1",
        )
        self.assertEqual(
            audit.OBJECTIVES,
            ("reconstruction", "envelope", "presence", "spectral_balance"),
        )
        self.assertIn("target_relative_presence_loss_v2", source)
        self.assertIn("mean_norms[\"reconstruction\"]", inspect.getsource(audit._derive_equalized_weights))
        self.assertIn("reference / norm", inspect.getsource(audit._derive_equalized_weights))
        self.assertIn("first_order_descent_dots", run_source)
        self.assertIn('"loss_weight_contract_authorized": False', run_source)
        self.assertIn('"new_vocoder_architecture_authorized": False', run_source)
        self.assertIn('"persistent_training_authorized": False', run_source)

    def test_audit_has_no_model_optimizer_or_checkpoint_training_path(self) -> None:
        source = inspect.getsource(audit).lower()
        for forbidden in (
            "torch.optim.",
            "optim.adam(",
            "optim.adamw(",
            ".backward(",
            ".step(",
            "torch.save(",
            "from_pretrained",
            "lykenoxvocodergenerator",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("torch.autograd.grad", source)


if __name__ == "__main__":
    unittest.main()
