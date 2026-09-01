from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_v4_2_replacement_decision as decision


class VocoderOwnershipContractTests(unittest.TestCase):
    def test_distribution_requires_lykenox_owned_architecture_and_weights(self) -> None:
        self.assertEqual(
            decision.VOCODER_OWNERSHIP_CONTRACT,
            "lykenox_owned_architecture_and_weights_only",
        )
        self.assertFalse(decision.THIRD_PARTY_PRETRAINED_VOCODER_AUTHORIZED)
        self.assertFalse(decision.THIRD_PARTY_VOCODER_CHECKPOINT_AUTHORIZED)
        self.assertTrue(decision.DISTRIBUTION_REQUIRES_LYKENOX_OWNED_WEIGHTS)

    def test_no_new_architecture_before_owned_pipeline_forensics(self) -> None:
        self.assertFalse(decision.NEW_VOCODER_ARCHITECTURE_AUTHORIZED)
        self.assertFalse(decision.SCRATCH_VOCODER_ITERATION_AUTHORIZED)
        self.assertEqual(
            decision.NEXT_ARCHITECTURE,
            "undecided_after_owned_pipeline_forensics",
        )
        self.assertEqual(
            decision.OWNED_VOCODER_DATA_CONTRACT,
            "vocoder-segment-v2-full-utterance-mel-pitch-conditioning",
        )
        self.assertEqual(decision.CONDITIONING_FORENSICS_STATUS, "pass")
        self.assertEqual(
            decision.NEXT_GATE,
            "audit_owned_vocoder_loss_edge_and_objective_semantics",
        )

    def test_decision_records_boundary_dominant_conditioning_mismatch(self) -> None:
        metrics = decision.CONDITIONING_FORENSIC_METRICS
        self.assertGreater(
            metrics["mean_boundary_f0_mae_cents_on_common_voiced"],
            metrics["mean_interior_f0_mae_cents_on_common_voiced"],
        )
        self.assertGreater(
            metrics["mean_boundary_periodicity_l1"],
            metrics["mean_interior_periodicity_l1"],
        )
        self.assertEqual(
            metrics["mean_interior_f0_mae_cents_on_common_voiced"],
            0.0,
        )

    def test_decision_contains_no_external_pretrained_replacement_route(self) -> None:
        source = inspect.getsource(decision).lower()
        self.assertNotIn("pretrained_vocoder_baseline", source)
        self.assertNotIn("charactr/vocos", source)
        self.assertNotIn("from_pretrained", source)


if __name__ == "__main__":
    unittest.main()
