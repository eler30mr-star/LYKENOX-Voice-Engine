from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer_audit as audit


class VocoderMinimumPhaseRendererAuditTests(unittest.TestCase):
    def test_audit_contract_blocks_model_training_and_checkpoints(self) -> None:
        self.assertEqual(audit.AUDIT_VERSION, "owned-minimum-phase-renderer-safety-audit-v1")
        self.assertEqual(audit.ARCHITECTURE_CONTRACT_VERSION, "owned-vocoder-architecture-contract-v1")
        self.assertFalse(audit.MODEL_INSTANTIATION_AUTHORIZED)
        self.assertFalse(audit.OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(audit.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(audit.NEW_VOCODER_CHECKPOINT_AUTHORIZED)

    def test_factorization_case_is_numerically_exact(self) -> None:
        result = audit._factorization_case()
        self.assertTrue(result["factorization_exact"])
        self.assertTrue(result["reference_oracle_representation_exact"])
        self.assertLess(result["maximum_log_magnitude_factorization_error"], 1e-10)
        self.assertLess(result["maximum_reference_oracle_roundtrip_error"], 1e-10)

    def test_identity_case_proves_exact_length_and_no_source_bypass(self) -> None:
        result = audit._identity_and_bypass_case()
        self.assertTrue(result["flat_envelope_exact_identity"])
        self.assertTrue(result["source_bypass_absent"])
        self.assertTrue(result["exact_output_length"])
        self.assertEqual(result["actual_sample_count"], result["expected_sample_count"])

    def test_audit_source_contains_no_trainable_or_checkpoint_path(self) -> None:
        source = inspect.getsource(audit).lower()
        for forbidden in (
            "nn.module",
            "torch.optim",
            ".backward(",
            ".step(",
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
