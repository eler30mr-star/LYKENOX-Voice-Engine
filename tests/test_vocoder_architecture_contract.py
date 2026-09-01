from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_architecture_contract as contract
from lykenox_voice_engine.training import speech_vocoder_loss_v2_weight_contract as weights


class VocoderArchitectureContractTests(unittest.TestCase):
    def test_selected_family_is_owned_minimum_phase_filter_renderer(self) -> None:
        self.assertEqual(
            contract.OWNED_VOCODER_ARCHITECTURE_CONTRACT_VERSION,
            "owned-vocoder-architecture-contract-v1",
        )
        self.assertEqual(contract.ARCHITECTURE_CONTRACT_VALIDATION_STATUS, "pass")
        self.assertEqual(contract.ARCHITECTURE_CONTRACT_VALIDATION_TEST_COUNT, 21)
        self.assertEqual(
            contract.STATIC_RENDERER_VERSION,
            "owned-minimum-phase-time-varying-renderer-v1",
        )
        self.assertTrue(contract.STATIC_RENDERER_SAFETY_AUDIT_REQUIRED)
        self.assertEqual(contract.STATIC_RENDERER_SAFETY_AUDIT_STATUS, "pending")
        self.assertEqual(
            contract.SELECTED_ARCHITECTURE_FAMILY,
            "owned_minimum_phase_time_varying_filter_over_neutral_excitation",
        )
        self.assertEqual(
            contract.TRAINABLE_REPRESENTATION,
            "frame_rate_real_cepstral_log_spectral_envelope",
        )
        self.assertEqual(contract.CEPSTRAL_ORDER, 64)
        self.assertEqual(contract.IDENTITY_CARRIER, "learned_spectral_envelope_filter_only")
        self.assertFalse(contract.EXCITATION_DIRECT_OUTPUT_BYPASS)

    def test_contract_uses_exact_owned_pipeline_and_frozen_weights(self) -> None:
        self.assertEqual(contract.SAMPLE_RATE, 24000)
        self.assertEqual(contract.HOP_LENGTH, 256)
        self.assertEqual(contract.N_FFT, 1024)
        self.assertEqual(contract.MEL_BINS, 80)
        self.assertEqual(contract.DATA_CONTRACT_VERSION, weights.DATA_CONTRACT_VERSION)
        self.assertEqual(contract.LOSS_CONTRACT_VERSION, weights.LOSS_CONTRACT_VERSION)
        self.assertEqual(contract.PRESENCE_CONTRACT_VERSION, weights.PRESENCE_CONTRACT_VERSION)
        self.assertEqual(
            contract.LOSS_WEIGHT_CONTRACT_VERSION,
            weights.OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION,
        )
        self.assertIn("f0_hz_from_full_utterance_pitch_cache", contract.CONDITIONING_INPUTS)
        self.assertIn("voiced_from_full_utterance_pitch_cache", contract.CONDITIONING_INPUTS)

    def test_historical_failure_mechanisms_are_forbidden(self) -> None:
        forbidden_flags = (
            "CONV_TRANSPOSE_UPSAMPLING_AUTHORIZED",
            "LEARNED_STRIDED_DECONVOLUTION_AUTHORIZED",
            "LEARNED_SAMPLE_RATE_UPSAMPLING_AUTHORIZED",
            "ABSOLUTE_COMPLEX_STFT_PREDICTION_AUTHORIZED",
            "INTEGRATED_FRAME_PHASE_RESIDUAL_AUTHORIZED",
            "DIRECT_SAMPLE_RATE_WAVEFORM_HEAD_AUTHORIZED",
            "DIRECT_WAVEFORM_RESIDUAL_BYPASS_AUTHORIZED",
            "DIRECT_HARMONIC_SINUSOID_BANK_OUTPUT_AUTHORIZED",
            "SOURCE_GATE_TO_WAVEFORM_BYPASS_AUTHORIZED",
            "CROP_LOCAL_PITCH_REANALYSIS_AUTHORIZED",
            "THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED",
        )
        for name in forbidden_flags:
            self.assertFalse(getattr(contract, name), name)

    def test_renderer_invariants_are_mandatory(self) -> None:
        self.assertTrue(contract.EXACT_OUTPUT_LENGTH_REQUIRED)
        self.assertEqual(
            contract.OUTPUT_LENGTH_RULE,
            "waveform_samples=conditioning_frames*256",
        )
        self.assertTrue(contract.FIXED_FRAME_TO_SAMPLE_INTERPOLATION_REQUIRED)
        self.assertTrue(contract.NEUTRAL_EXCITATION_MUST_HAVE_NO_LEARNED_IDENTITY_PARAMETERS)
        self.assertTrue(contract.FILTER_MUST_BE_MINIMUM_PHASE_BY_CONSTRUCTION)
        self.assertTrue(contract.SOURCE_MUST_NOT_REACH_OUTPUT_UNFILTERED)
        self.assertTrue(contract.HOP_GRID_CARRIER_REJECTION_REQUIRED)
        self.assertTrue(contract.DOUBLE_HOP_GRID_CARRIER_REJECTION_REQUIRED)
        self.assertTrue(contract.FLAT_ENVELOPE_RENDERER_IDENTITY_TEST_REQUIRED)
        self.assertTrue(contract.REFERENCE_ENVELOPE_ORACLE_DIAGNOSTIC_REQUIRED)
        self.assertFalse(contract.REFERENCE_ENVELOPE_ORACLE_PRODUCT_PATH_AUTHORIZED)

    def test_contract_does_not_authorize_model_or_training(self) -> None:
        self.assertTrue(contract.STATIC_RENDERER_IMPLEMENTATION_AUTHORIZED)
        self.assertFalse(contract.MODEL_INSTANTIATION_AUTHORIZED)
        self.assertFalse(contract.OPTIMIZER_CREATION_AUTHORIZED)
        self.assertFalse(contract.PERSISTENT_TRAINING_AUTHORIZED)
        self.assertFalse(contract.NEW_VOCODER_CHECKPOINT_AUTHORIZED)
        self.assertFalse(contract.METRICS_CAN_ACCEPT_PRODUCT_QUALITY)
        self.assertTrue(contract.FULL_HELD_OUT_AUDIO_REQUIRED_FOR_PRODUCT_ACCEPTANCE)
        self.assertEqual(
            contract.NEXT_GATE,
            "prove_owned_minimum_phase_renderer_factorization_length_and_grid_safety_before_model",
        )

    def test_contract_has_no_model_or_training_implementation(self) -> None:
        source = inspect.getsource(contract).lower()
        for forbidden in (
            "torch.nn",
            "nn.module",
            "torch.optim",
            ".backward(",
            ".step(",
            "from_pretrained",
            "convtranspose",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
