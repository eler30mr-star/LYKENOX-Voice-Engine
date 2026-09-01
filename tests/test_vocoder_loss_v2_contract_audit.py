from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_loss_v2_contract_audit as audit


class OwnedVocoderLossV2ContractAuditTests(unittest.TestCase):
    def test_audit_contract_is_read_only_and_blocks_model_work(self) -> None:
        source = inspect.getsource(audit).lower()
        run_source = inspect.getsource(audit.run_owned_vocoder_loss_v2_contract_audit)
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-vocoder-loss-v2-target-consistency-audit-v1",
        )
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
        self.assertIn('"persistent_training_authorized": False', run_source)
        self.assertIn('"new_vocoder_architecture_authorized": False', run_source)
        self.assertIn('"third_party_model_used": False', run_source)

    def test_audit_requires_target_and_conditioning_zero_consistency(self) -> None:
        source = inspect.getsource(audit.run_owned_vocoder_loss_v2_contract_audit)
        self.assertIn("target_reconstruction_exact", source)
        self.assertIn("conditioning_envelope_exact", source)
        self.assertIn("exact_frame_contract", source)
        self.assertIn("TARGET_ZERO_TOLERANCE", source)
        self.assertIn(
            '"audit_owned_vocoder_loss_v2_gradient_balance_before_architecture_selection"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
