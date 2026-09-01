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

    def test_only_architecture_weight_recalibration_is_open(self) -> None:
        self.assertTrue(decision.LOSS_WEIGHT_CONTRACT_AUTHORIZED)
        self.assertTrue(decision.LOSS_V2_WEIGHT_CONTRACT_FROZEN)
        self.assertFalse(decision.LOSS_V2_WEIGHT_CONTRACT_V1_ARCHITECTURE_COMPATIBLE)
        self.assertFalse(decision.ACTIVE_ARCHITECTURE_WEIGHT_CONTRACT_AUTHORIZED)
        self.assertFalse(decision.NEW_VOCODER_ARCHITECTURE_AUTHORIZED)
        self.assertFalse(decision.SCRATCH_VOCODER_ITERATION_AUTHORIZED)
        self.assertTrue(decision.FRAME_RATE_CEPSTRAL_PREDICTOR_IMPLEMENTATION_AUTHORIZED)
        self.assertTrue(decision.MODEL_INSTANTIATION_AUTHORIZED)
        self.assertEqual(decision.BOUNDED_OPTIMIZER_SMOKE_STATUS, "pass")
        self.assertTrue(decision.BOUNDED_OPTIMIZER_SMOKE_CONSUMED)
        self.assertFalse(decision.BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED)
        self.assertEqual(decision.PARAMETER_SPACE_GRADIENT_AUDIT_STATUS, "pass")
        self.assertFalse(decision.PARAMETER_SPACE_GRADIENT_AUDIT_AUTHORIZED)
        self.assertTrue(decision.ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_AUTHORIZED)
        self.assertFalse(decision.ARCHITECTURE_WEIGHT_CONTRACT_V2_AUTHORIZED)
        self.assertFalse(decision.EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED)
        self.assertFalse(decision.OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(decision.TRAINER_IMPLEMENTATION_AUTHORIZED)
        self.assertFalse(decision.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(decision.NEW_VOCODER_CHECKPOINT_AUTHORIZED)
        self.assertTrue(decision.VOCODER_ARCHITECTURE_SELECTION_AUTHORIZED)
        self.assertEqual(
            decision.NEXT_ARCHITECTURE,
            "owned_minimum_phase_time_varying_filter_over_neutral_excitation",
        )
        self.assertEqual(
            decision.NEXT_GATE,
            "audit_owned_minimum_phase_architecture_coupled_loss_v2_weight_recalibration",
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
        self.assertEqual(
            decision.OWNED_VOCODER_ARCHITECTURE_CONTRACT,
            "owned-vocoder-architecture-contract-v1",
        )
        self.assertEqual(
            decision.OWNED_STATIC_RENDERER,
            "owned-minimum-phase-time-varying-renderer-v1",
        )
        self.assertEqual(
            decision.OWNED_FRAME_RATE_PREDICTOR,
            "lykenox_owned_frame_rate_cepstral_predictor_v1",
        )
        self.assertEqual(
            decision.ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_VERSION,
            "owned-minimum-phase-architecture-weight-recalibration-audit-v1",
        )
        self.assertEqual(
            decision.ARCHITECTURE_WEIGHT_RECALIBRATION_CANDIDATE_VERSION,
            "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2-candidate",
        )
        self.assertTrue(decision.HISTORICAL_PRESENCE_EDGE_SEMANTICS_REJECTED)

    def test_all_completed_gates_are_recorded_pass(self) -> None:
        self.assertEqual(decision.CONDITIONING_FORENSICS_STATUS, "pass")
        self.assertEqual(decision.LOSS_EDGE_FORENSICS_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_TARGET_CONSISTENCY_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_GRADIENT_BALANCE_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_FOUR_OBJECTIVE_CALIBRATION_STATUS, "pass")
        self.assertEqual(decision.LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_STATUS, "pass")
        self.assertFalse(decision.LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_REQUIRED)
        self.assertEqual(decision.ARCHITECTURE_CONTRACT_STATUS, "pass")
        self.assertEqual(decision.ARCHITECTURE_CONTRACT_VALIDATION_TEST_COUNT, 21)
        self.assertEqual(decision.STATIC_RENDERER_SAFETY_STATUS, "pass")
        self.assertEqual(decision.STATIC_RENDERER_SAFETY_TEST_COUNT, 24)
        self.assertEqual(decision.FRAME_RATE_PREDICTOR_STRUCTURAL_STATUS, "pass")
        self.assertEqual(decision.FRAME_RATE_PREDICTOR_STRUCTURAL_TEST_COUNT, 36)
        self.assertEqual(decision.BOUNDED_OPTIMIZER_SMOKE_STATUS, "pass")
        self.assertEqual(decision.BOUNDED_OPTIMIZER_SMOKE_TEST_COUNT, 33)
        self.assertEqual(decision.PARAMETER_SPACE_GRADIENT_AUDIT_STATUS, "pass")
        self.assertEqual(decision.PARAMETER_SPACE_GRADIENT_AUDIT_TEST_COUNT, 41)
        self.assertEqual(decision.PARAMETER_SPACE_GRADIENT_AUDIT_PROBE_COUNT, 8)

    def test_bounded_optimizer_evidence_triggered_jacobian_review(self) -> None:
        metrics = decision.BOUNDED_OPTIMIZER_SMOKE_METRICS
        self.assertEqual(metrics["update_count"], 2)
        self.assertLess(metrics["final_total"], metrics["initial_total"])
        self.assertGreater(metrics["final_envelope"], metrics["initial_envelope"])
        self.assertGreater(metrics["update_1_raw_gradient_norm"], 500.0)
        self.assertGreater(metrics["update_2_raw_gradient_norm"], 500.0)
        self.assertEqual(metrics["parameter_delta_norm"], 0.0004)
        self.assertFalse(metrics["severe_grid_excess"])
        self.assertTrue(metrics["checkpoints_unchanged"])

    def test_parameter_jacobian_audit_rejects_v1_for_active_architecture(self) -> None:
        metrics = decision.PARAMETER_SPACE_GRADIENT_AUDIT_METRICS
        parameter_shares = metrics["neutral_mean_weighted_gradient_norm_shares"]
        self.assertGreater(parameter_shares["spectral_balance"], 0.70)
        self.assertLess(parameter_shares["envelope"], 0.01)
        self.assertLess(
            metrics["neutral_minimum_combined_gradient_alignment_cosines"]["envelope"],
            0.0,
        )
        self.assertLess(
            metrics["neutral_minimum_first_order_descent_dots"]["envelope"],
            0.0,
        )
        self.assertGreater(metrics["neutral_mean_combined_gradient_norm"], 500.0)
        self.assertLess(metrics["neutral_mean_clip_scale_if_max_norm_1"], 0.01)
        self.assertGreater(
            metrics["cepstrum_neutral_mean_weighted_gradient_norm_shares"]["spectral_balance"],
            0.75,
        )
        self.assertLess(
            metrics["cepstrum_neutral_mean_weighted_gradient_norm_shares"]["envelope"],
            0.01,
        )
        self.assertLess(metrics["cepstrum_neutral_minimum_envelope_alignment"], 0.0)
        self.assertLess(metrics["cepstrum_connected_minimum_envelope_alignment"], 0.0)
        self.assertFalse(decision.LOSS_V2_WEIGHT_CONTRACT_V1_ARCHITECTURE_COMPATIBLE)
        self.assertTrue(decision.ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_AUTHORIZED)
        self.assertFalse(decision.EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED)
        self.assertFalse(decision.PERSISTENT_TRAINING_AUTHORIZED)

    def test_predictor_structural_evidence_remains_valid(self) -> None:
        metrics = decision.FRAME_RATE_PREDICTOR_STRUCTURAL_METRICS
        self.assertEqual(metrics["predictor_output_shape"], (2, 48, 64))
        self.assertEqual(metrics["maximum_abs_initial_cepstrum"], 0.0)
        self.assertEqual(metrics["renderer_identity_max_abs_error"], 0.0)
        self.assertEqual(metrics["expected_waveform_samples"], 12288)
        self.assertEqual(metrics["actual_waveform_samples"], 12288)
        self.assertEqual(metrics["connected_nonzero_gradient_tensor_count"], 30)
        self.assertEqual(metrics["trainable_parameter_tensor_count"], 30)

    def test_renderer_safety_evidence_remains_valid(self) -> None:
        metrics = decision.STATIC_RENDERER_SAFETY_METRICS
        self.assertLess(metrics["maximum_log_magnitude_factorization_error"], 1e-10)
        self.assertLess(metrics["maximum_reference_oracle_roundtrip_error"], 1e-10)
        self.assertEqual(metrics["flat_envelope_max_abs_identity_error"], 0.0)
        self.assertTrue(metrics["source_bypass_absent"])
        self.assertTrue(metrics["exact_output_length"])
        self.assertLess(abs(metrics["unvoiced_hop_autocorrelation_excess"]), 0.01)
        self.assertLess(abs(metrics["voiced_hop_autocorrelation_excess"]), 0.01)

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
        self.assertEqual(metrics["mean_interior_f0_mae_cents_on_common_voiced"], 0.0)

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
            self.assertEqual(metrics[f"{prefix}_mean_interior_log_magnitude_l1"], 0.0)

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

    def test_v1_weights_remain_frozen_as_historical_waveform_space_contract(self) -> None:
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
        self.assertTrue(sensitivity["candidate_tracks_derivation"])
        self.assertTrue(sensitivity["authority_retained"])
        self.assertTrue(sensitivity["alignment_positive"])
        self.assertTrue(sensitivity["descent_positive"])
        self.assertTrue(sensitivity["dominance_bounded"])
        self.assertFalse(decision.LOSS_V2_WEIGHT_CONTRACT_V1_ARCHITECTURE_COMPATIBLE)
        self.assertFalse(decision.ACTIVE_ARCHITECTURE_WEIGHT_CONTRACT_AUTHORIZED)

    def test_decision_contains_no_external_pretrained_replacement_route(self) -> None:
        source = inspect.getsource(decision).lower()
        self.assertNotIn("pretrained_vocoder_baseline", source)
        self.assertNotIn("charactr/vocos", source)
        self.assertNotIn("from_pretrained", source)


if __name__ == "__main__":
    unittest.main()
