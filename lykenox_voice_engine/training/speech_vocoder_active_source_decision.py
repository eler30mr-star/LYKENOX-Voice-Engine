"""Authoritative active engineering decision for the LYKENOX vocoder source path.

This file records the cross-variant direct-reference and common-conditioning evidence obtained on
2026-09-03. New source architectures/training remain frozen while the shared conditioning contract
is corrected and validated. Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v13-conditioning-v2-gate"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

# Learned source history remains rejection/forensic evidence.
CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PITCH_SYNCHRONOUS_CYCLE_SPEECH_0021_NEAR_REFERENCE_VOICE = True
PHASE_CONTINUOUS_DECODER_STATUS = "retained_for_forensics_but_insufficient"
PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
UNIFIED_SOURCE_RUN_STATUS = "rejected_same_audible_defects_persist"
UNIFIED_SOURCE_BEST_VAL_TOTAL = 4.632137616475423
UNIFIED_SOURCE_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.695, 0.728, 0.740)
UNIFIED_SOURCE_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

# Direct reference comparison over 95 already-generated final WAVs. The clean real-residual and
# identity-roundtrip ceilings remain closest to reference. Learned sources share excessive tonal
# prominence and upper-band deficits. This diagnostic localizes/rejects only.
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_STATUS = "complete"
CROSS_VARIANT_REFERENCE_COMPARISON_COUNT = 95
DIRECT_REFERENCE_CLOSEST_DIAGNOSTIC = "real_residual_resynthesis"
DIRECT_REFERENCE_REAL_RESIDUAL_SCORE = 3.4813
DIRECT_REFERENCE_REAL_RESIDUAL_RMS_DELTA_DB = 0.05
DIRECT_REFERENCE_CONTINUOUS_V2_SCORE = 7.3656
DIRECT_REFERENCE_CONTINUOUS_V2_TONAL_PROMINENCE_EXCESS_P95_DB = 7.62
UNIFIED_SOURCE_DIRECT_REFERENCE_TONAL_PROMINENCE_EXCESS_P95_DB = (7.88, 6.15, 8.54)

# Cross-variant common anomaly locations.
SHARED_ANOMALY_SPEECH_0021_TIME_SECONDS = 5.80
SHARED_ANOMALY_SPEECH_0021_VARIANT_COUNT = 10
SHARED_ANOMALY_SPEECH_0021_MEAN_TONAL_EXCESS_DB = 8.574
SHARED_ANOMALY_SPEECH_0021_MEAN_AIR_EXCESS_DB = 7.651
SHARED_ANOMALY_SPEECH_0024_PRIMARY_TIME_SECONDS = 4.00
SHARED_ANOMALY_SPEECH_0024_PRIMARY_VARIANT_COUNT = 13
SHARED_ANOMALY_SPEECH_0024_PRIMARY_MEAN_TONAL_EXCESS_DB = 16.576
SHARED_ANOMALY_SPEECH_0024_SECONDARY_TIME_SECONDS = 5.20
SHARED_ANOMALY_SPEECH_0024_SECONDARY_VARIANT_COUNT = 12

# Common-conditioning forensic result.
# speech_0024 ~4.00 s: F0=86.96 Hz, voiced=1, periodicity=0.481, neighboring F0 jump ratio=1.025,
# no near-octave competitor, yet tonal excess=16.576 dB. Therefore octave jumping is NOT the main
# explanation and transition conditioning alone cannot explain all tonal artifacts.
SPEECH_0024_4S_F0_STABLE = True
SPEECH_0024_4S_NEAR_OCTAVE_COMPETITOR = False
SPEECH_0024_4S_TONAL_EXCESS_DB = 16.576

# speech_0024 ~5.20 s: F0=0, voiced=0, raw periodicity=0.233 and six voiced-state changes nearby.
# speech_0021 ~5.80 s: F0=0, voiced=0, raw periodicity=0.416, RMS only 3.4% of utterance peak,
# with tonal/air excess shared by ten variants. These frames expose a contract inconsistency: pitch-v1
# hard-zeros F0 and voiced while preserving raw autocorrelation periodicity as an independent feature.
SPEECH_0024_5S2_VOICED_STATE_CHANGES = 6
SPEECH_0021_5S8_RAW_PERIODICITY = 0.416
SPEECH_0021_5S8_RMS_FRACTION_OF_PEAK = 0.034
PITCH_V1_TRANSITION_SEMANTICS_REJECTED = True
PITCH_V1_OCTAVE_ERROR_IS_PRIMARY_ROOT_CAUSE = False

# Source-code audit confirms the semantic contradiction reaches learned models directly:
# continuous V2 concatenates log-F0, binary voiced and raw periodicity as independent conditioning;
# unified does the same, then additionally uses voiced*periodicity for periodic energy and stops phase
# whenever voiced<0.5. Thus F0=0 / voiced=0 / periodicity>0 is not merely a diagnostic artifact.
PITCH_V1_CONTRACT = "f0_zero_when_unvoiced_plus_binary_voiced_plus_raw_periodicity"
PITCH_V1_CONTRACT_IS_COHERENT_FOR_SOURCE_TRANSITIONS = False

# Conditioning v2 corrects the shared semantics without changing trusted anchor pitch values:
# - trusted pitch-v1 voiced-anchor F0 is preserved exactly;
# - non-anchor gaps use continuous log-F0 interpolation/edge hold rather than F0 reset to zero;
# - binary voiced is removed from product conditioning authority;
# - raw periodicity is converted to energy-aware continuous periodic_strength using the same v1 RMS
#   threshold parameter in smooth form;
# - energy_confidence is exposed explicitly.
PITCH_CONDITIONING_V2 = "lykenox-pitch-conditioning-v2-continuous-strength"
PITCH_CONDITIONING_V2_MODULE = "lykenox_voice_engine/training/speech_pitch_conditioning_v2.py"
PITCH_CONDITIONING_V2_PRESERVES_TRUSTED_ANCHOR_F0 = True
PITCH_CONDITIONING_V2_HARD_F0_ZERO_RESET = False
PITCH_CONDITIONING_V2_BINARY_VOICED_PRODUCT_AUTHORITY = False
PITCH_CONDITIONING_V2_ENERGY_AWARE_PERIODIC_STRENGTH = True
PITCH_CONDITIONING_V2_VALIDATION_SCRIPT = "scripts/run_pitch_conditioning_v2_anomaly_audit.py"

# This is a contract/root correction, not an output patch. It generates no audio and applies no gain,
# EQ, denoise or duration change. It is not yet authorized for source retraining until its behavior at
# the already-known 12 anomaly events is validated.
ACTIVE_ENGINEERING_GATE = "validate_pitch_conditioning_v2_at_existing_common_anomaly_events"
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False

# After the v2 conditioning audit, only the transition-related part of the defect may be attributed to
# this contract. The stable voiced ~4.00 s anomaly remains a separate source-generation problem unless
# corrected conditioning evidence shows otherwise. No claim of full root-cause closure is authorized.
NEXT_ACTION = "run_pitch_conditioning_v2_common_anomaly_audit_before_any_source_retraining"
