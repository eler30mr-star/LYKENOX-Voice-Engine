from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training import speech_vocoder_loss_v2_gradient_balance_audit as audit
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
)


class OwnedVocoderLossV2GradientBalanceAuditTests(unittest.TestCase):
    def test_audit_is_waveform_gradient_only_and_cannot_train_models(self) -> None:
        source = inspect.getsource(audit).lower()
        run_source = inspect.getsource(
            audit.run_owned_vocoder_loss_v2_gradient_balance_audit
        )
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-vocoder-loss-v2-gradient-balance-audit-v1",
        )
        self.assertIn("torch.autograd.grad", source)
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
        self.assertIn('"training_started": False', run_source)
        self.assertIn('"optimizer_created": False', run_source)
        self.assertIn('"model_instantiated": False', run_source)
        self.assertIn('"persistent_training_authorized": False', run_source)
        self.assertIn('"new_vocoder_architecture_authorized": False', run_source)
        self.assertIn('"loss_weight_contract_authorized": False', run_source)

    def test_diagnostic_candidates_are_exact_length_and_nontrivial(self) -> None:
        torch.manual_seed(17)
        target = torch.randn(1, 64 * 256, dtype=torch.float32) * 0.05
        candidates = audit._diagnostic_candidates(target, sample_rate=24000)
        self.assertEqual(
            set(candidates),
            {"v4_2_like_color", "low_band_excess", "one_sample_phase_smear"},
        )
        for candidate in candidates.values():
            self.assertEqual(candidate.shape, target.shape)
            self.assertTrue(bool(torch.isfinite(candidate).all()))
            self.assertGreater(float((candidate - target).abs().mean()), 0.0)

    def test_objective_probe_produces_finite_nonzero_waveform_gradients(self) -> None:
        torch.manual_seed(23)
        config = LykenoxSpeechConfig()
        target = torch.randn(1, 64 * config.hop_length, dtype=torch.float32) * 0.05
        envelope = ConditioningAlignedLogMelEnvelopeLossV2(config)
        with torch.no_grad():
            conditioning = envelope._generated_log_mel(target)[:, :64, :].contiguous()
            candidate = audit._diagnostic_candidates(
                target,
                sample_rate=config.sample_rate,
            )["v4_2_like_color"]
        probe = audit._objective_probe(
            candidate,
            target,
            conditioning,
            envelope,
            sample_rate=config.sample_rate,
        )
        self.assertTrue(probe["gradients_finite_nonzero"])
        self.assertTrue(probe["combined_gradient_finite_nonzero"])
        self.assertGreater(float(probe["combined_gradient_norm"]), 0.0)
        shares = probe["reference_weighted_gradient_norm_shares"]
        self.assertAlmostEqual(sum(float(value) for value in shares.values()), 1.0, places=5)
        for value in probe["pairwise_gradient_cosines"].values():
            self.assertTrue(-1.00001 <= float(value) <= 1.00001)
        for value in probe["combined_gradient_alignment_cosines"].values():
            self.assertTrue(-1.00001 <= float(value) <= 1.00001)


if __name__ == "__main__":
    unittest.main()
