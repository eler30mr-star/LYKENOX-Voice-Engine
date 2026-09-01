"""Owned LYKENOX vocoder architecture contract after Jacobian authority review.

The owned architecture family, fixed minimum-phase renderer, frame-rate cepstral predictor,
and exactly-two-update real-data optimizer smoke passed their bounded structural/trainability
gates.  The subsequent read-only parameter/Jacobian audit also passed as a measurement, but
it proved that the waveform-space Loss V2 weight contract v1 is not authority-compatible
with this minimum-phase representation: spectral balance dominates after the renderer,
while envelope authority collapses and can become anti-aligned with the combined direction.

Therefore v1 remains frozen as valid waveform-space evidence but is explicitly forbidden as
the training-weight contract for this architecture.  The only open gate is a read-only
architecture-coupled recalibration derived in cepstrum space and cross-checked in predictor
parameter space.  No optimizer, extended smoke, trainer, persistent training, checkpoint
creation, or product-quality acceptance is authorized.
"""

from __future__ import annotations


OWNED_VOCODER_ARCHITECTURE_CONTRACT_VERSION = "owned-vocoder-architecture-contract-v1"
ARCHITECTURE_CONTRACT_VALIDATION_STATUS = "pass"
ARCHITECTURE_CONTRACT_VALIDATION_TEST_COUNT = 21

STATIC_RENDERER_VERSION = "owned-minimum-phase-time-varying-renderer-v1"
STATIC_RENDERER_SAFETY_AUDIT_REQUIRED = True
STATIC_RENDERER_SAFETY_AUDIT_VERSION = "owned-minimum-phase-renderer-safety-audit-v1"
STATIC_RENDERER_SAFETY_AUDIT_STATUS = "pass"
STATIC_RENDERER_SAFETY_AUDIT_TEST_COUNT = 24
STATIC_RENDERER_SAFETY_EVIDENCE = {
    "maximum_log_magnitude_factorization_error": 1.0685896612017132e-15,
    "maximum_reference_oracle_roundtrip_error": 6.938893903907228e-18,
    "flat_envelope_max_abs_identity_error": 0.0,
    "attenuating_filter_measured_rms_ratio": 0.002478752176666358,
    "attenuating_filter_max_abs_expected_error": 8.673617379884035e-19,
    "unvoiced_hop_autocorrelation_excess": -0.001309794233417115,
    "unvoiced_double_hop_autocorrelation_excess": -0.0013493497067419598,
    "unvoiced_grid_harmonic_power_fraction_excess": 0.00012359877649464196,
    "voiced_hop_autocorrelation_excess": 9.394839821898096e-05,
    "voiced_double_hop_autocorrelation_excess": 0.0005020093189166902,
    "voiced_grid_harmonic_power_fraction_excess": 0.00040792484893605215,
    "same_seed_max_abs_error": 0.0,
}

FRAME_RATE_PREDICTOR_ARCHITECTURE = "lykenox_owned_frame_rate_cepstral_predictor_v1"
FRAME_RATE_PREDICTOR_STRUCTURAL_SMOKE_VERSION = (
    "owned-frame-rate-cepstral-predictor-smoke-v1"
)
FRAME_RATE_PREDICTOR_STRUCTURAL_SMOKE_STATUS = "pass"
FRAME_RATE_PREDICTOR_STRUCTURAL_SMOKE_TEST_COUNT = 36
FRAME_RATE_PREDICTOR_STRUCTURAL_EVIDENCE = {
    "predictor_output_shape": (2, 48, 64),
    "maximum_abs_initial_cepstrum": 0.0,
    "renderer_identity_max_abs_error": 0.0,
    "expected_waveform_samples": 12288,
    "actual_waveform_samples": 12288,
    "hop_autocorrelation_excess": 0.0,
    "double_hop_autocorrelation_excess": 0.0,
    "grid_harmonic_power_fraction_excess": 0.0,
    "parameter_count": 236736,
    "connected_nonzero_gradient_tensor_count": 30,
    "trainable_parameter_tensor_count": 30,
}

BOUNDED_OPTIMIZER_SMOKE_VERSION = "owned-minimum-phase-bounded-optimizer-smoke-v1"
BOUNDED_OPTIMIZER_SMOKE_STATUS = "pass"
BOUNDED_OPTIMIZER_SMOKE_TEST_COUNT = 33
BOUNDED_OPTIMIZER_SMOKE_CONSUMED = True
BOUNDED_OPTIMIZER_SMOKE_EVIDENCE = {
    "segment_mel_frames": 32,
    "max_items": 1,
    "max_updates": 2,
    "learning_rate": 0.0002,
    "max_gradient_norm": 1.0,
    "initial_total": 195.1538543701,
    "final_total": 194.9214477539,
    "relative_total_change": -0.0011908892,
    "initial_reconstruction": 19.2587738037,
    "final_reconstruction": 19.2581729889,
    "initial_envelope": 4.7642612457,
    "final_envelope": 4.7644839287,
    "initial_presence": 3.1395702362,
    "final_presence": 3.1374154091,
    "initial_spectral_balance": 1.6438173056,
    "final_spectral_balance": 1.640686512,
    "update_1_raw_gradient_norm": 580.735168457,
    "update_2_raw_gradient_norm": 580.9564819336,
    "parameter_delta_norm": 0.0004,
    "parameter_delta_max_abs": 0.0000964731,
    "final_hop_autocorrelation_excess": -0.00002784491516649723,
    "final_double_hop_autocorrelation_excess": 0.00002341344952583313,
    "final_grid_harmonic_power_fraction_excess": 0.000048296526074409485,
    "checkpoints_unchanged": True,
}
BOUNDED_OPTIMIZER_TOTAL_DESCENT_CONFIRMED = True
BOUNDED_OPTIMIZER_CLIP_REGIME_REVIEW_REQUIRED = True
BOUNDED_OPTIMIZER_ENVELOPE_LOCAL_INCREASE_OBSERVED = True

PARAMETER_SPACE_GRADIENT_AUDIT_VERSION = (
    "owned-minimum-phase-parameter-gradient-authority-audit-v1"
)
PARAMETER_SPACE_GRADIENT_AUDIT_STATUS = "pass"
PARAMETER_SPACE_GRADIENT_AUDIT_TEST_COUNT = 41
PARAMETER_SPACE_GRADIENT_AUDIT_PROBE_COUNT = 8
PARAMETER_SPACE_GRADIENT_AUDIT_EVIDENCE = {
    "neutral_mean_weighted_gradient_norm_shares": {
        "reconstruction": 0.046928433266268124,
        "envelope": 0.009529865430589276,
        "presence": 0.234379802425534,
        "spectral_balance": 0.7091618988776086,
    },
    "neutral_minimum_combined_gradient_alignment_cosines": {
        "reconstruction": 0.08097482472658157,
        "envelope": -0.3273468315601349,
        "presence": 0.8204574584960938,
        "spectral_balance": 0.976812481880188,
    },
    "neutral_minimum_first_order_descent_dots": {
        "reconstruction": 478.39349365234375,
        "envelope": -387.89593505859375,
        "presence": 2462.26708984375,
        "spectral_balance": 2270.551025390625,
    },
    "neutral_mean_combined_gradient_norm": 524.2026596069336,
    "neutral_mean_clip_scale_if_max_norm_1": 0.0019496183077621803,
    "neutral_maximum_weighted_gradient_norm_share": 0.765470299656555,
    "connected_mean_weighted_gradient_norm_shares": {
        "reconstruction": 0.046930590150991255,
        "envelope": 0.009529655236837993,
        "presence": 0.2343777644546192,
        "spectral_balance": 0.7091619901575515,
    },
    "connected_minimum_envelope_combined_alignment": -0.32732290029525757,
    "connected_minimum_envelope_descent_dot": -387.87164306640625,
    "connected_mean_combined_gradient_norm": 524.2039108276367,
    "connected_mean_clip_scale_if_max_norm_1": 0.00194961708097689,
    "connected_maximum_weighted_gradient_norm_share": 0.7654694679457218,
}
CEPSTRUM_SPACE_GRADIENT_AUDIT_EVIDENCE = {
    "neutral_mean_weighted_gradient_norm_shares": {
        "reconstruction": 0.04117481310162209,
        "envelope": 0.008032201568982572,
        "presence": 0.1949435194527072,
        "spectral_balance": 0.7558494658766881,
    },
    "neutral_minimum_envelope_combined_alignment": -0.2579599916934967,
    "connected_minimum_envelope_combined_alignment": -0.2580244839191437,
}
WAVEFORM_SPACE_WEIGHT_CONTRACT_VERSION = "owned-vocoder-loss-v2-weight-contract-v1"
WAVEFORM_SPACE_WEIGHT_CONTRACT_FROZEN = True
WAVEFORM_SPACE_WEIGHT_CONTRACT_VALID_IN_ORIGINAL_CALIBRATION_SPACE = True
WAVEFORM_SPACE_WEIGHT_CONTRACT_ARCHITECTURE_COMPATIBLE = False
ARCHITECTURE_WEIGHT_RECALIBRATION_REQUIRED = True
ARCHITECTURE_WEIGHT_RECALIBRATION_DERIVATION_SPACE = "cepstrum_space"
ARCHITECTURE_WEIGHT_RECALIBRATION_CROSS_CHECK_SPACE = "parameter_space"
ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_VERSION = (
    "owned-minimum-phase-architecture-weight-recalibration-audit-v1"
)
ARCHITECTURE_WEIGHT_CONTRACT_V2_AUTHORIZED = False

SELECTED_ARCHITECTURE_FAMILY = (
    "owned_minimum_phase_time_varying_filter_over_neutral_excitation"
)

DATA_CONTRACT_VERSION = "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
LOSS_CONTRACT_VERSION = "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
PRESENCE_CONTRACT_VERSION = "owned-vocoder-presence-v2-valid-context-target-relative"
LOSS_WEIGHT_CONTRACT_VERSION = WAVEFORM_SPACE_WEIGHT_CONTRACT_VERSION

SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_FFT = 1024
MEL_BINS = 80
CEPSTRAL_ORDER = 64

CONDITIONING_INPUTS = (
    "mel_80",
    "f0_hz_from_full_utterance_pitch_cache",
    "voiced_from_full_utterance_pitch_cache",
    "periodicity_from_full_utterance_pitch_cache",
)

TRAINABLE_REPRESENTATION = "frame_rate_real_cepstral_log_spectral_envelope"
FIXED_ENVELOPE_TO_FILTER_TRANSFORM = "real_cepstrum_to_minimum_phase_fir"
FIXED_EXCITATION = "bandlimited_impulse_train_plus_deterministic_aperiodic_noise"
FIXED_RENDERING = "time_varying_minimum_phase_filtering_with_fixed_crossfade"
IDENTITY_CARRIER = "learned_spectral_envelope_filter_only"
EXCITATION_DIRECT_OUTPUT_BYPASS = False

# Historical mechanisms rejected by measured failures.
CONV_TRANSPOSE_UPSAMPLING_AUTHORIZED = False
LEARNED_STRIDED_DECONVOLUTION_AUTHORIZED = False
LEARNED_SAMPLE_RATE_UPSAMPLING_AUTHORIZED = False
ABSOLUTE_COMPLEX_STFT_PREDICTION_AUTHORIZED = False
INTEGRATED_FRAME_PHASE_RESIDUAL_AUTHORIZED = False
DIRECT_SAMPLE_RATE_WAVEFORM_HEAD_AUTHORIZED = False
DIRECT_WAVEFORM_RESIDUAL_BYPASS_AUTHORIZED = False
DIRECT_HARMONIC_SINUSOID_BANK_OUTPUT_AUTHORIZED = False
SOURCE_GATE_TO_WAVEFORM_BYPASS_AUTHORIZED = False
CROP_LOCAL_PITCH_REANALYSIS_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False

# Renderer invariants remain mandatory for every later gate.
EXACT_OUTPUT_LENGTH_REQUIRED = True
OUTPUT_LENGTH_RULE = "waveform_samples=conditioning_frames*256"
FIXED_FRAME_TO_SAMPLE_INTERPOLATION_REQUIRED = True
NEUTRAL_EXCITATION_MUST_HAVE_NO_LEARNED_IDENTITY_PARAMETERS = True
FILTER_MUST_BE_MINIMUM_PHASE_BY_CONSTRUCTION = True
FILTER_ENVELOPE_MUST_BE_FRAME_RATE = True
SOURCE_MUST_NOT_REACH_OUTPUT_UNFILTERED = True
HOP_GRID_CARRIER_REJECTION_REQUIRED = True
DOUBLE_HOP_GRID_CARRIER_REJECTION_REQUIRED = True
FLAT_ENVELOPE_RENDERER_IDENTITY_TEST_REQUIRED = True
REFERENCE_ENVELOPE_ORACLE_DIAGNOSTIC_REQUIRED = True

REFERENCE_ENVELOPE_ORACLE_PRODUCT_PATH_AUTHORIZED = False
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELD_OUT_AUDIO_REQUIRED_FOR_PRODUCT_ACCEPTANCE = True

# Completed optimizer/Jacobian gates are closed. Only read-only v2 weight recalibration opens.
STATIC_RENDERER_IMPLEMENTATION_AUTHORIZED = True
FRAME_RATE_CEPSTRAL_PREDICTOR_IMPLEMENTATION_AUTHORIZED = True
MODEL_INSTANTIATION_AUTHORIZED = True
BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED = False
PARAMETER_SPACE_GRADIENT_AUDIT_AUTHORIZED = False
ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_AUTHORIZED = True
EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED = False
OPTIMIZER_CREATION_AUTHORIZED = False
TRAINER_IMPLEMENTATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False

NEXT_GATE = (
    "audit_owned_minimum_phase_architecture_coupled_loss_v2_weight_recalibration"
)
