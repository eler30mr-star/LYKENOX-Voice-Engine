"""Owned LYKENOX vocoder architecture contract after predictor structural proof.

The owned architecture family, fixed minimum-phase renderer, and frame-rate cepstral
predictor have passed their structural gates.  This contract now authorizes exactly one
additional capability: an ephemeral, tightly bounded optimizer smoke over owned V2 data.
It still does not authorize a trainer, persistent training, checkpoint creation, or
product-quality acceptance.

The selected family keeps pitch timing information while excluding the historical shortcuts
that allowed a carrier, learned upsampler, or frame-phase representation to dominate the
waveform. Voice identity must be carried by a learned time-varying spectral envelope/filter
trained only from LYKENOX-owned data. Excitation remains fixed, spectrally neutral, and has
no direct output bypass.
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
    "owned-frame-rate-cepstral-predictor-structural-smoke-v1"
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
SELECTED_ARCHITECTURE_FAMILY = (
    "owned_minimum_phase_time_varying_filter_over_neutral_excitation"
)

DATA_CONTRACT_VERSION = "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
LOSS_CONTRACT_VERSION = "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
PRESENCE_CONTRACT_VERSION = "owned-vocoder-presence-v2-valid-context-target-relative"
LOSS_WEIGHT_CONTRACT_VERSION = "owned-vocoder-loss-v2-weight-contract-v1"

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

# Renderer invariants remain mandatory for every model/optimizer smoke.
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

# Scoped authorization: exactly one bounded smoke may create an ephemeral optimizer.
STATIC_RENDERER_IMPLEMENTATION_AUTHORIZED = True
FRAME_RATE_CEPSTRAL_PREDICTOR_IMPLEMENTATION_AUTHORIZED = True
MODEL_INSTANTIATION_AUTHORIZED = True
BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED = True
BOUNDED_OPTIMIZER_SMOKE_MAX_UPDATES = 2
BOUNDED_OPTIMIZER_SMOKE_SEGMENT_FRAMES = 32
BOUNDED_OPTIMIZER_SMOKE_MAX_ITEMS = 1
OPTIMIZER_CREATION_AUTHORIZED = False  # general/trainer optimizer creation remains blocked
TRAINER_IMPLEMENTATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False

NEXT_GATE = (
    "prove_owned_predictor_real_data_bounded_optimizer_descent_without_grid_or_checkpoint_regression"
)
