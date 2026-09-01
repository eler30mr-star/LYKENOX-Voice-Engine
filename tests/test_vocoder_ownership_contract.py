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

    def test_loss_contract_is_frozen_but_model_work_stays_blocked(self) -> None:
        self.assertTrue(decision.LOSS_WEIGHT_CONTRACT_AUTHORIZED)
        self.assertTrue(decision.LOSS_V2_WEIGHT_CONTRACT_FROZEN)
        self.assertFalse(decision.NEW_VOCODER_ARCHITECTURE_AUTHORIZED)
        self.assertFalse(decision.SCRATCH_VOCODER_ITERATION_AUTHORIZED)
        self.assertFalse(decision.MODEL_INSTANTIATION_AUTHORIZED)
        self.assertFalse(decision.OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(decision.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertTrue(decision.VOCODER_ARCHITECTURE_SELECTION_AUTHORIZED)
        self.assertEqual(
            decision.NEXT_ARCHITECTURE,
            "undecided_owned_architecture_after_loss_v2_weight_contract",
        )
        self.assertEqual(
            decision.NEXT_GATE,
            "define_owned_vocoder_architecture_contract_before_model_instantiation",
        )

    def test_owned_pipeline_contracts_are_exact(self) -> None:
        self.assertEqual(
            decision.OWNED_VOCODER_DATA_CONTRACT,
            "vocoder-segment-v2-full-utterance-mel-pitch-conditioning",
        )
        self.assertEqual(
            decision.OWNED_VOCODER_LOSS_CONTRACT,
            "owned-vocoder-loss-v2-valid-context-conditioning-aligned",
        )
        self.assertEqual(
            decision.OWNED_VOCODER_PRESENCE_CONTRACT,
            "owned-vocoder-presence-v2-valid-context-target-relative",
        )
        self.assertEqual(
            decision.OWNED_VOCODER_LOSS_WEIGHT_CONTRACT,
            "owned-vocoder-loss-v2-weight-contract-v1",
        )
        self.assertTrue(decision.HISTORICAL_PRESENCE_EDGE_SEMANTICS_REJECTED)

    def test_forensic_and_loss_gates_are_all_recorded_pass(self) -> None:
        self.assertEqual(decision.CONDITIONING_FORENSICS_STATUS, "pass")
        self.assertEqual(decision.LOSS_EDGE_FORENSICS_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_TARGET_CONSISTENCY_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_GRADIENT_BALANCE_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_FOUR_OBJECTIVE_CALIBRATION_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_STATUS, "pass")
        self.assertFalse(decision.LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_REQUIRED)

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

    def test_decision_records_artificial_loss_edge_bug_and_64_frame_contract(self) -> None:
        metrics = decision.LOSS_EDGE_FORENSIC_METRICS
        self.assertEqual(metrics["mel_crop_local_frame_count"], 65)
        self.assertEqual(metrics["mel_conditioning_frame_count"], 64)
        self.assertTrue(metrics["mel_extra_terminal_frame_without_conditioning"])
        self.assertGreater(
            metrics["mean_mel_artificial_log_l1"],
            metrics["mean_mel_interior_log_l1"],
        )
        for prefix in ("stft_256_64", "stft_512_128", "stft_1024_256"):
            self.assertGreater(
                metrics[f"{prefix}_mean_artificial_log_magnitude_l1"],
                metrics[f"{prefix}_mean_interior_log_magnitude_l1"],
            )
            self.assertEqual(
                metrics[f"{prefix}_mean_interior_log_magnitude_l1"],
                0.0,
            )

    def test_loss_v2_target_consistency_remains_exact(self) -> None:
        metrics = decision.LOSS_V2_TARGET_CONSISTENCY_METRICS
        self.assertTrue(metrics["exact_conditioning_frame_contract"])
        self.assertTrue(metrics["target_reconstruction_exact_on_valid_context"])
        self.assertTrue(metrics["conditioning_envelope_exact_on_valid_context"])
        self.assertEqual(metrics["conditioning_frames"], 64)
        self.assertEqual(metrics["analysis_frames"], 65)
        self.assertEqual(metrics["valid_conditioning_frames"], 61)
        self.assertEqual(metrics["reconstruction_valid_frame_counts"], (253, 125, 61))
        self.assertEqual(metrics["reconstruction_analysis_frame_counts"], (257, 129, 65))
        self.assertLess(metrics["mean_conditioning_aligned_envelope_total"], 1e-6)

    def test_historical_reference_weights_remain_rejected(self) -> None:
        metrics = decision.LOSS_V2_GRADIENT_BALANCE_METRICS
        self.assertGreater(metrics["mean_reference_weighted_reconstruction_share"], 0.80)
        self.assertLess(metrics["mean_reference_weighted_spectral_balance_share"], 0.01)
        self.assertGreater(metrics["maximum_reference_weighted_gradient_norm_share"], 0.90)
        self.assertNotEqual(
            decision.LOSS_V2_WEIGHT_CONTRACT_FROZEN_WEIGHTS,
            {
                "reconstruction": 1.0,
                "envelope": 0.50,
                "presence": 0.0,
                "spectral_balance": 0.25,
            },
        )

    def test_four_objective_weights_are_frozen_after_sensitivity_pass(self) -> None:
        self.assertEqual(
            decision.LOSS_V2_WEIGHT_CONTRACT_FROZEN_WEIGHTS,
            {
                "reconstruction": 1.0,
                "envelope": 3.1475,
                "presence": 19.3369,
                "spectral_balance": 60.9496,
            },
        )
        sensitivity = decision.LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_METRICS
        self.assertEqual(sensitivity["scenario_count"], 23)
        self.assertEqual(sensitivity["relative_weight_perturbation"], 0.10)
        self.assertTrue(sensitivity["candidate_tracks_derivation"])
        self.assertTrue(sensitivity["authority_retained"])
        self.assertTrue(sensitivity["alignment_positive"])
        self.assertTrue(sensitivity["alignment_retained"])
        self.assertTrue(sensitivity["descent_positive"])
        self.assertTrue(sensitivity["dominance_bounded"])
        self.assertLess(
            sensitivity["all_scenarios_maximum_weighted_gradient_norm_share"],
            0.60,
        )
        for value in sensitivity[
            "all_scenarios_minimum_combined_gradient_alignment_cosines"
        ].values():
            self.assertGreater(value, 0.0)
        for value in sensitivity[
            "all_scenarios_minimum_first_order_descent_dots"
        ].values():
            self.assertGreater(value, 0.0)

    def test_decision_contains_no_external_pretrained_replacement_route(self) -> None:
        source = inspect.getsource(decision).lower()
        self.assertNotIn("pretrained_vocoder_baseline", source)
        self.assertNotIn("charactr/vocos", source)
        self.assertNotIn("from_pretrained", source)


if __name__ == "__main__":
    unittest.main()
