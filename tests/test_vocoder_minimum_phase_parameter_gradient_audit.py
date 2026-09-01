from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training import (
    speech_vocoder_minimum_phase_parameter_gradient_audit as audit,
)


class VocoderMinimumPhaseParameterGradientAuditTests(unittest.TestCase):
    def test_audit_scope_is_read_only_and_real_data_bounded(self) -> None:
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-minimum-phase-parameter-gradient-authority-audit-v1",
        )
        self.assertEqual(audit.SEGMENT_MEL_FRAMES, 32)
        self.assertEqual(audit.ITEMS_PER_SPLIT, 2)
        self.assertEqual(audit.SPLITS, ("train", "val"))
        self.assertEqual(audit.STATES, ("neutral", "connected_probe"))
        self.assertEqual(audit.CONNECTED_HEAD_SCALE, 1.0e-4)
        self.assertEqual(audit.REFERENCE_MAX_GRAD_NORM, 1.0)
        self.assertFalse(audit.OPTIMIZER_CREATED)
        self.assertFalse(audit.PARAMETER_UPDATE_EXECUTED)
        self.assertFalse(audit.TRAINER_INSTANTIATED)
        self.assertFalse(audit.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(audit.NEW_VOCODER_CHECKPOINT_AUTHORIZED)
        self.assertFalse(audit.EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED)

    def test_space_metrics_measure_frozen_weight_authority_and_direction(self) -> None:
        gradients = {
            "reconstruction": torch.tensor([1.0, 0.0], dtype=torch.float64),
            "envelope": torch.tensor([0.5, 0.5], dtype=torch.float64),
            "presence": torch.tensor([0.0, 1.0], dtype=torch.float64),
            "spectral_balance": torch.tensor([0.25, 0.75], dtype=torch.float64),
        }
        metrics = audit._space_metrics(gradients)
        shares = metrics["weighted_gradient_norm_shares"]
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=12)
        self.assertGreater(metrics["combined_gradient_norm"], 0.0)
        self.assertGreater(metrics["clip_scale_if_max_norm_1"], 0.0)
        self.assertLessEqual(metrics["clip_scale_if_max_norm_1"], 1.0)
        self.assertTrue(metrics["all_objective_gradients_finite_nonzero"])
        self.assertTrue(metrics["combined_gradient_finite_nonzero"])
        for value in metrics["first_order_descent_dots"].values():
            self.assertGreater(value, 0.0)

    def test_combined_gradient_linearity_check_is_exact_for_matching_direct_gradient(self) -> None:
        gradients = {
            "reconstruction": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "envelope": torch.tensor([0.5, -0.2], dtype=torch.float64),
            "presence": torch.tensor([-0.1, 0.8], dtype=torch.float64),
            "spectral_balance": torch.tensor([0.3, 0.1], dtype=torch.float64),
        }
        weights = audit.FROZEN_WEIGHTS.as_dict()
        direct = sum(
            gradients[name] * float(weights[name]) for name in audit.OBJECTIVES
        )
        metrics = audit._space_metrics(gradients, direct_combined=direct)
        self.assertLessEqual(
            metrics["combined_gradient_linearity_relative_error"],
            1e-12,
        )

    def test_audit_uses_owned_pipeline_renderer_and_frozen_objectives(self) -> None:
        source = inspect.getsource(audit)
        for required in (
            "collect_owned_vocoder_segments",
            "render_owned_minimum_phase_vocoder_path",
            "valid_context_multi_resolution_reconstruction_loss",
            "ConditioningAlignedLogMelEnvelopeLossV2",
            "target_relative_presence_loss_v2",
            "target_relative_spectral_balance_loss",
            "combine_owned_vocoder_loss_v2",
            "FROZEN_WEIGHTS",
            "torch.autograd.grad",
        ):
            self.assertIn(required, source)

    def test_audit_cannot_create_optimizer_train_or_write_checkpoint(self) -> None:
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
        self.assertIn('"persistent_training_authorized": false', source)
        self.assertIn('"extended_trainability_smoke_authorized": false', source)


if __name__ == "__main__":
    unittest.main()
