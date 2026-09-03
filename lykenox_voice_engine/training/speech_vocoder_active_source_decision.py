"""Authoritative active engineering decision for the LYKENOX vocoder source path.

This file records the direct-reference, common-conditioning and pitch-conditioning-v2 evidence
obtained on 2026-09-03. No new source architecture is authorized. Exactly one controlled retrain of
the historical Continuous Residual Source V2 architecture is authorized to isolate the verified
conditioning-contract correction. Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v14-one-controlled-conditioning-v2-retrain"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

# Historical perceptual/source evidence.
CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PITCH_SYNCHRONOUS_CYCLE_SPEECH_0021_NEAR_REFERENCE_VOICE = True
PHASE_CONTINUOUS_DECODER_STATUS = "retained_for_forensics_but_insufficient"
PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
UNIFIED_SOURCE_RUN_STATUS = "rejected_same_audible_defects_persist"
UNIFIED_SOURCE_BEST_VAL_TOTAL = 4.632137616475423
UNIFIED_SOURCE_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.695, 0.728, 0.740)

# Direct generated-vs-reference evidence across 95 existing final WAVs.
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_STATUS = "complete"
CROSS_VARIANT_REFERENCE_COMPARISON_COUNT = 95
DIRECT_REFERENCE_CLOSEST_DIAGNOSTIC = "real_residual_resynthesis"
DIRECT_REFERENCE_REAL_RESIDUAL_SCORE = 3.4813
DIRECT_REFERENCE_REAL_RESIDUAL_RMS_DELTA_DB = 0.05
DIRECT_REFERENCE_CONTINUOUS_V2_SCORE = 7.3656
DIRECT_REFERENCE_CONTINUOUS_V2_TONAL_PROMINENCE_EXCESS_P95_DB = 7.62
SHARED_ANOMALY_SPEECH_0021_TIME_SECONDS = 5.80
SHARED_ANOMALY_SPEECH_0021_VARIANT_COUNT = 10
SHARED_ANOMALY_SPEECH_0021_MEAN_TONAL_EXCESS_DB = 8.574
SHARED_ANOMALY_SPEECH_0021_MEAN_AIR_EXCESS_DB = 7.651
SHARED_ANOMALY_SPEECH_0024_PRIMARY_TIME_SECONDS = 4.00
SHARED_ANOMALY_SPEECH_0024_PRIMARY_VARIANT_COUNT = 13
SHARED_ANOMALY_SPEECH_0024_PRIMARY_MEAN_TONAL_EXCESS_DB = 16.576
SHARED_ANOMALY_SPEECH_0024_SECONDARY_TIME_SECONDS = 5.20
SHARED_ANOMALY_SPEECH_0024_SECONDARY_VARIANT_COUNT = 12

# Common-conditioning forensics rejected octave jumping as the primary explanation. The strongest
# common transition defect is semantic inconsistency in pitch-v1: F0 is hard-zeroed and voiced is
# binary while raw autocorrelation periodicity remains nonzero and is delivered independently to all
# learned sources.
PITCH_V1_TRANSITION_SEMANTICS_REJECTED = True
PITCH_V1_OCTAVE_ERROR_IS_PRIMARY_ROOT_CAUSE = False
PITCH_V1_CONTRACT = "f0_zero_when_unvoiced_plus_binary_voiced_plus_raw_periodicity"
PITCH_V1_CONTRACT_IS_COHERENT_FOR_SOURCE_TRANSITIONS = False

# Stable voiced tonal defect remains separate. At speech_0024 ~4.00 s the cached F0 is stable at
# 86.96 Hz, voiced=1 and periodicity=0.481 with no near-octave competitor, while learned variants show
# about +16.576 dB shared tonal-prominence excess. Conditioning repair is not allowed to claim this
# stable-voiced source defect is solved unless the new held-out audio actually demonstrates it.
SPEECH_0024_4S_F0_STABLE = True
SPEECH_0024_4S_NEAR_OCTAVE_COMPETITOR = False
SPEECH_0024_4S_TONAL_EXCESS_DB = 16.576
STABLE_VOICED_TONAL_EXCESS_REMAINS_SEPARATE_OPEN_DEFECT = True

# Pitch-conditioning V2 audit completed over the 12 already-localized common anomaly events.
PITCH_CONDITIONING_V2 = "lykenox-pitch-conditioning-v2-continuous-strength"
PITCH_CONDITIONING_V2_MODULE = "lykenox_voice_engine/training/speech_pitch_conditioning_v2.py"
PITCH_CONDITIONING_V2_AUDIT_STATUS = "complete_partial_root_correction_confirmed"
PITCH_CONDITIONING_V2_AUDIT_EVENT_COUNT = 12
PITCH_V1_CONTRADICTORY_UNVOICED_PERIODICITY_EVENT_COUNT = 3
PITCH_CONDITIONING_V2_F0_ZERO_RESET_REMOVED_EVENT_COUNT = 8
PITCH_CONDITIONING_V2_PRESERVES_TRUSTED_ANCHOR_F0 = True
PITCH_CONDITIONING_V2_HARD_F0_ZERO_RESET = False
PITCH_CONDITIONING_V2_BINARY_VOICED_PRODUCT_AUTHORITY = False
PITCH_CONDITIONING_V2_ENERGY_AWARE_PERIODIC_STRENGTH = True

# Key audit events.
SPEECH_0021_5S8_V1 = {"f0_hz": 0.0, "voiced": 0.0, "periodicity": 0.416}
SPEECH_0021_5S8_V2 = {"f0_track_hz": 185.78, "periodic_strength": 0.123, "energy_confidence": 0.297}
SPEECH_0024_5S2_V1 = {"f0_hz": 0.0, "voiced": 0.0, "periodicity": 0.233}
SPEECH_0024_5S2_V2 = {"f0_track_hz": 117.12, "periodic_strength": 0.174, "energy_confidence": 0.747}
SPEECH_0024_5S2_V2_LOCAL_F0_JUMP_RATIO = 2.03
SPEECH_0024_4S_V2 = {"f0_track_hz": 86.96, "periodic_strength": 0.433, "energy_confidence": 0.901}
SPEECH_0024_4S_V2_LOCAL_F0_JUMP_RATIO = 1.44

# Interpretation: conditioning V2 is a verified correction to transition semantics, not a complete
# root-cause closure. It removes hard F0 resets and greatly reduces low-energy periodic authority at
# known transition failures, but local F0 discontinuities remain in some contexts and the stable
# voiced ~4.00 s tonal anomaly is not explained by transition semantics.
PITCH_CONDITIONING_V2_FULL_ROOT_CAUSE_CLAIM_AUTHORIZED = False
PITCH_CONDITIONING_V2_TRANSITION_ROOT_CORRECTION_CONFIRMED = True

# Exactly one controlled retrain is authorized to determine whether this verified upstream correction
# materially improves held-out audio. Continuous Residual Source V2 is selected because it is a full
# source without the rejected pitch-sync/V2 handoff and provides a clean historical baseline. This is
# NOT a new vocoder architecture.
ACTIVE_ENGINEERING_GATE = "one_controlled_continuous_source_v2_retrain_with_pitch_conditioning_v2"
CONTROLLED_RETRAIN_AUTHORIZED = True
CONTROLLED_RETRAIN_COUNT_AUTHORIZED = 1
CONTROLLED_RETRAIN_ARCHITECTURE = "lykenox_owned_continuous_residual_source_v2_level_factored"
CONTROLLED_RETRAIN_TRAINER = "lykenox_voice_engine/training/speech_vocoder_continuous_residual_source_v2_pitch_conditioning_v2.py"
CONTROLLED_RETRAIN_ENTRYPOINT = "scripts/train_continuous_residual_source_v2_pitch_conditioning_v2.py"
CONTROLLED_RETRAIN_RENDERER = "scripts/render_continuous_residual_source_v2_pitch_conditioning_v2.py"
CONTROLLED_RETRAIN_RUN_DIR = "models/lykenox_identity/training/continuous_residual_source_v2_pitch_conditioning_v2"
CONTROLLED_RETRAIN_ARCHITECTURE_CHANGED = False
CONTROLLED_RETRAIN_LOSS_CHANGED = False
CONTROLLED_RETRAIN_STEP3F_TARGET_CHANGED = False
CONTROLLED_RETRAIN_RENDERER_CHANGED = False
CONTROLLED_RETRAIN_TRAINING_BUDGET_CHANGED = False
CONTROLLED_RETRAIN_BASELINE_WARM_START = "models/lykenox_identity/training/continuous_residual_source_v2/best.pt"
CONTROLLED_RETRAIN_ONLY_FUNCTIONAL_CHANGE = "pitch_conditioning_v1_to_v2"
CONTROLLED_RETRAIN_HELDOUT_COMPARATORS = (
    "v2_pitch_conditioning_v2",
    "v2_baseline_source",
    "identity_roundtrip_ceiling",
    "reference",
)

# The controlled run is allowed to reject or support the transition-conditioning correction only.
# If audible transition chirp/wind/nasal artifacts do not materially decrease versus the historical
# V2 baseline, no further source retraining is authorized from this hypothesis. Even if transitions
# improve, product acceptance still requires separate resolution of stable voiced tonal excess.
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_BEYOND_CONTROLLED_RETRAIN_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False

NEXT_ACTION = "run_exactly_one_controlled_continuous_source_v2_retrain_with_pitch_conditioning_v2_then_listen_against_historical_v2_baseline_and_identity_ceiling"
