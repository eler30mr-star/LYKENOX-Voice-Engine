"""Authoritative active source-path decision for the LYKENOX vocoder.

Historical codebook and source implementations remain evidence. This file states the active
engineering gate after the 2026-09-03 cross-variant direct reference comparison.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v12-shared-conditioning-forensics"

# The most recent learned source is retained only as forensic evidence. Human listening reports that
# the same unacceptable audible defects remain, and direct WAV/reference comparison shows strong
# anomaly timestamps shared across many otherwise different learned-source architectures. No new
# source architecture or training is authorized until the shared conditioning is audited at those
# exact timestamps.
ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_unified_phase_residual_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_minimum_phase_unified_phase_residual_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_unified_phase_residual_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_unified_phase_residual_source_v1.py"
ACTIVE_SOURCE_CHECKPOINT = "models/lykenox_identity/training/unified_phase_residual_source_v1/best.pt"
ACTIVE_SOURCE_STATUS = "rejected_same_audible_defects_persist"
ACTIVE_ENGINEERING_GATE = "shared_conditioning_forensics_at_cross_variant_common_anomaly_times"
ACTIVE_DIAGNOSTIC = "scripts/diagnose_vocoder_common_conditioning_anomalies_v1.py"
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = False

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
SAMPLE_RATE_AUTOREGRESSIVE_MODEL = False

CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V1_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.098, 0.128, 0.153)
CONTINUOUS_SOURCE_V1_CHECKPOINT_MAY_BE_USED_FOR_PRODUCT = False

CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
CONTINUOUS_SOURCE_V2_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
COHERENT_INNOVATION_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False
FURTHER_COHERENT_INNOVATION_TUNING_AUTHORIZED = False

# Pitch-synchronous real Step3f residual cycles produced the strongest learned voiced source so far.
# On speech_0021 the owner reported the V1 pitch-synchronous waveform as almost like the original,
# with a remaining robotic whistle/chirp. This remains important evidence but not a complete source.
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PITCH_SYNCHRONOUS_CYCLE_UPDATES = 600
PITCH_SYNCHRONOUS_CYCLE_BEST_VAL_TOTAL = 2.6395082473754883
PITCH_SYNCHRONOUS_CYCLE_SPEECH_0021_NEAR_REFERENCE_VOICE = True
PITCH_SYNCHRONOUS_CYCLE_CHECKPOINT_RETAINED_FOR_FORENSICS = True
PITCH_SYNCHRONOUS_CYCLE_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

PITCH_SYNCHRONOUS_V1_HARD_SPLICE_DECODER_REJECTED = True
PHASE_CONTINUOUS_DECODER_STATUS = "retained_for_forensics_but_insufficient"
FURTHER_HARD_SPLICE_DECODER_USE_AUTHORIZED = False
FURTHER_PHASE_CONTINUOUS_DECODER_TUNING_AUTHORIZED = False

PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
PHASE_EXCLUSIVE_HANDOFF_TRAINING_EXECUTED = False
PHASE_EXCLUSIVE_HANDOFF_RAW_SAMPLEWISE_MIX_USED = False
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_NASALITY = True
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_WIND_NOISE = True
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_SLIGHT_ROBOTIZATION = True
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_RESIDUAL_CHIRP = True
FURTHER_SOURCE_HANDOFF_OR_BRIDGE_TUNING_AUTHORIZED = False
HYBRID_V2_PLUS_PITCH_SYNC_PRODUCT_PATH_CLOSED = True

UNIFIED_SOURCE_RUN_STATUS = "rejected_same_audible_defects_persist"
UNIFIED_SOURCE_UPDATES = 600
UNIFIED_SOURCE_BEST_VAL_TOTAL = 4.632137616475423
UNIFIED_SOURCE_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.695, 0.728, 0.740)
UNIFIED_SOURCE_HELDOUT_LEVEL_DEFICIT_PRESENT = True
UNIFIED_SOURCE_SINGLE_MODEL = True
UNIFIED_SOURCE_SINGLE_RECURRENT_STATE = True
UNIFIED_SOURCE_SOURCE_HANDOFF_OR_BRIDGE_USED = False
UNIFIED_SOURCE_CODEBOOK_USED = False
UNIFIED_SOURCE_TEACHER_FORCING_USED = False
UNIFIED_SOURCE_STOCHASTIC_INNOVATION_USED = False
UNIFIED_SOURCE_POSTHOC_GAIN_NORMALIZATION_USED = False
UNIFIED_SOURCE_POSTHOC_EQ_USED = False
UNIFIED_SOURCE_POSTHOC_DENOISING_USED = False
UNIFIED_SOURCE_METRICS_ACCEPT_PRODUCT_QUALITY = False
UNIFIED_SOURCE_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

# Direct generated-vs-reference comparison completed over the already-generated evaluation corpus.
# This diagnostic generates no audio, executes no training or model inference, and cannot accept
# product quality. It established that the real-residual and identity-roundtrip ceilings are the
# closest outputs, while every learned source remains materially separated from reference.
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_STATUS = "complete"
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_VERSION = "owned-vocoder-generated-vs-reference-diagnostic-v1"
CROSS_VARIANT_REFERENCE_COMPARISON_COUNT = 95
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_GENERATES_AUDIO = False
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_RUNS_MODEL_INFERENCE = False
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_WRITES_CHECKPOINT = False
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_USES_IDENTITY_CEILING_AS_METRIC_FLOOR = True
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False

# Diagnostic ranking is localization evidence only, not acceptance. The real-residual resynthesis and
# identity-roundtrip resynthesis are nearest to reference, consistent with human listening. Learned
# source outputs show excess tonal prominence and reduced upper-band energy.
DIRECT_REFERENCE_CLOSEST_DIAGNOSTIC = "real_residual_resynthesis"
DIRECT_REFERENCE_REAL_RESIDUAL_SCORE = 3.4813
DIRECT_REFERENCE_REAL_RESIDUAL_RMS_DELTA_DB = 0.05
DIRECT_REFERENCE_COHERENT_INNOVATION_SCORE = 4.7646
DIRECT_REFERENCE_CONTINUOUS_V2_SCORE = 7.3656
DIRECT_REFERENCE_CONTINUOUS_V2_TONAL_PROMINENCE_EXCESS_P95_DB = 7.62
DIRECT_REFERENCE_CONTINUOUS_V2_HIGH_BAND_ENERGY_DELTA_DB = -1.33

UNIFIED_SOURCE_DIRECT_REFERENCE_RMS_RATIOS = (0.695, 0.728, 0.740)
UNIFIED_SOURCE_DIRECT_REFERENCE_RMS_DELTA_DB = (-3.16, -2.76, -2.61)
UNIFIED_SOURCE_DIRECT_REFERENCE_TONAL_PROMINENCE_EXCESS_P95_DB = (7.88, 6.15, 8.54)
UNIFIED_SOURCE_DIRECT_REFERENCE_HIGH_BAND_DELTA_DB = (-1.58, -2.05, -1.89)
UNIFIED_SOURCE_DIRECT_REFERENCE_AIR_BAND_DELTA_DB = (-2.23, -2.36, -2.78)

# Strong anomaly timestamps recur across many independently designed learned source variants. This is
# the key new localization evidence. It argues against another source-architecture change before the
# common conditioning contract is inspected at the same frames.
SHARED_ANOMALY_SPEECH_0021_TIME_SECONDS = 5.80
SHARED_ANOMALY_SPEECH_0021_VARIANT_COUNT = 10
SHARED_ANOMALY_SPEECH_0021_MEAN_TONAL_EXCESS_DB = 8.57
SHARED_ANOMALY_SPEECH_0021_MEAN_AIR_EXCESS_DB = 7.65
SHARED_ANOMALY_SPEECH_0024_PRIMARY_TIME_SECONDS = 4.00
SHARED_ANOMALY_SPEECH_0024_PRIMARY_VARIANT_COUNT = 13
SHARED_ANOMALY_SPEECH_0024_PRIMARY_MEAN_TONAL_EXCESS_DB = 16.57
SHARED_ANOMALY_SPEECH_0024_SECONDARY_TIME_SECONDS = 5.20
SHARED_ANOMALY_SPEECH_0024_SECONDARY_VARIANT_COUNT = 12
SHARED_ANOMALY_SPEECH_0024_SECONDARY_CHARACTER = "rms_drop_and_high_spectral_error"

# All active learned source families consume the same owned mel/F0/voiced/periodicity full-utterance
# conditioning contract. real_residual_resynthesis and identity_roundtrip do not need to predict the
# residual source and remain clean. pitch-v1 currently selects the strongest autocorrelation lag in
# each frame independently and uses an utterance-relative RMS threshold for voiced state. This is a
# plausible common upstream failure mechanism at low-energy/word-boundary frames, but it is NOT yet
# accepted as the root cause. The dedicated diagnostic must inspect raw top autocorrelation lags,
# octave competitors, F0 jumps, periodicity and voiced transitions at the shared anomaly timestamps.
ACTIVE_ROOT_CAUSE_HYPOTHESIS = "shared_pitch_voicing_conditioning_error_at_common_anomaly_frames"
ACTIVE_ROOT_CAUSE_HYPOTHESIS_CONFIRMED = False
ACTIVE_ROOT_CAUSE_HYPOTHESIS_REQUIRES_CONDITIONING_FORENSICS = True
COMMON_CONDITIONING_PITCH_CACHE_VERSION = "speech-pitch-cache-v1"
COMMON_CONDITIONING_PITCH_TARGET_VERSION = "lykenox-pitch-v1"
COMMON_CONDITIONING_DIAGNOSTIC_VERSION = "owned-vocoder-common-conditioning-anomaly-diagnostic-v1"
COMMON_CONDITIONING_DIAGNOSTIC_GENERATES_AUDIO = False
COMMON_CONDITIONING_DIAGNOSTIC_RUNS_MODEL_INFERENCE = False
COMMON_CONDITIONING_DIAGNOSTIC_WRITES_CHECKPOINT = False

PITCH_SYNCHRONOUS_REAL_CYCLE_EXTRACTION_AVAILABLE = True
PITCH_SYNCHRONOUS_REAL_CYCLE_SOURCE_IS_PARAMETRIC_ROSENBERG = False
DISCRETE_RESIDUAL_CODEBOOK_PRODUCT_PATH_CLOSED = True
CELP_CODEBOOK_SELECTOR_TRAINING_AUTHORIZED = False
FURTHER_CODEBOOK_RETENTION_SWEEPS_AUTHORIZED = False
FURTHER_CODEBOOK_PRESELECTION_TUNING_AUTHORIZED = False
FURTHER_CODEBOOK_BEAM_TUNING_AUTHORIZED = False
PARAMETRIC_ROSENBERG_SOURCE_REOPEN_AUTHORIZED = False

THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False

TRAIN_SPLIT_ONLY_FOR_OPTIMIZER_UPDATES = True
VAL_ALLOWED_FOR_REJECTION_AND_CHECKPOINT_SELECTION = True
COMPLETE_HELDOUT_FREE_RUNNING_VALIDATION_REQUIRED = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

NEXT_ACTION = "run_common_conditioning_anomaly_diagnostic_before_any_new_source_training_or_architecture_change"
