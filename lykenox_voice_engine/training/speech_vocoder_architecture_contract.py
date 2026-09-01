"""Owned LYKENOX vocoder architecture contract before any model instantiation.

This contract selects one architecture *family* for the next static renderer gate. It does
not instantiate a neural network, create an optimizer, start training, or authorize a new
checkpoint. The family is chosen from failures already measured in LYKENOX V4.2/V7/V8/V9
and from the validated owned data/loss/weight contracts.

The next candidate keeps pitch timing information but removes the historical shortcuts that
let a carrier, learned upsampler, or frame-phase representation dominate the waveform. Voice
identity must be carried by a learned time-varying spectral envelope/filter trained from
LYKENOX-owned data. The excitation is required to be spectrally neutral and has no direct
output bypass.
"""

from __future__ import annotations


OWNED_VOCODER_ARCHITECTURE_CONTRACT_VERSION = "owned-vocoder-architecture-contract-v1"
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

# The trainable path is frame-rate only. It may predict a smooth real-cepstral spectral
# envelope, but it may not directly predict waveform samples or phase-bearing STFT bins.
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

# Required renderer invariants before any neural model class may be instantiated.
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

# The reference-envelope oracle is structural only: it may prove the renderer can express
# the target envelope, but it is never a product inference path and cannot accept quality.
REFERENCE_ENVELOPE_ORACLE_PRODUCT_PATH_AUTHORIZED = False
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELD_OUT_AUDIO_REQUIRED_FOR_PRODUCT_ACCEPTANCE = True

# Model/training remain blocked until the fixed renderer itself passes the next gate.
STATIC_RENDERER_IMPLEMENTATION_AUTHORIZED = True
MODEL_INSTANTIATION_AUTHORIZED = False
OPTIMIZER_CREATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False

NEXT_GATE = (
    "prove_owned_minimum_phase_renderer_factorization_length_and_grid_safety_before_model"
)
