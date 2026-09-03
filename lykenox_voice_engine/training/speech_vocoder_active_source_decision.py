"""Authoritative active source-path decision for the LYKENOX vocoder.

Historical codebook and source implementations remain evidence. This file states the active
engineering path after the 2026-09-03 phase-exclusive handoff listening regression.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v9-unified-phase-residual-source"

ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_unified_phase_residual_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_minimum_phase_unified_phase_residual_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_unified_phase_residual_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_unified_phase_residual_source_v1.py"
ACTIVE_ONE_COMMAND_ENTRYPOINT = "scripts/train_unified_phase_residual_source_v1.py"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
SAMPLE_RATE_AUTOREGRESSIVE_MODEL = False

CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V1_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.098, 0.128, 0.153)
CONTINUOUS_SOURCE_V1_CHECKPOINT_MAY_BE_USED_FOR_PRODUCT = False

# V2 fixed the amplitude collapse and remains a useful historical source baseline, but its held-out
# voice remained robotic. It may not be used as a final product source or as an independent fallback
# branch in the active architecture.
CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
CONTINUOUS_SOURCE_V2_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
COHERENT_INNOVATION_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False
FURTHER_COHERENT_INNOVATION_TUNING_AUTHORIZED = False

# Pitch-synchronous real Step3f residual cycles produced the strongest learned voiced source so far.
# On speech_0021 the owner reported the V1 pitch-synchronous waveform as almost like the original,
# with a remaining robotic whistle/chirp. This proves the useful representation is F0-relative phase,
# not fixed 512/256 absolute sample phase.
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PITCH_SYNCHRONOUS_CYCLE_UPDATES = 600
PITCH_SYNCHRONOUS_CYCLE_BEST_VAL_TOTAL = 2.6395082473754883
PITCH_SYNCHRONOUS_CYCLE_SPEECH_0021_NEAR_REFERENCE_VOICE = True
PITCH_SYNCHRONOUS_CYCLE_CHECKPOINT_RETAINED_FOR_FORENSICS = True
PITCH_SYNCHRONOUS_CYCLE_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

# Hard-splice decoding was structurally invalid and phase-continuous Fourier decoding removed that
# defect, but the terminal chirp remained. No further cycle-decoder tuning is authorized.
PITCH_SYNCHRONOUS_V1_HARD_SPLICE_DECODER_REJECTED = True
PHASE_CONTINUOUS_DECODER_STATUS = "retained_for_forensics_but_insufficient"
FURTHER_HARD_SPLICE_DECODER_USE_AUTHORIZED = False
FURTHER_PHASE_CONTINUOUS_DECODER_TUNING_AUTHORIZED = False

# Phase-exclusive V3 replaced raw source crossfade with pitch-sync authority inside complete cycles,
# V2 authority elsewhere, and a period-derived C1 Hermite bridge at every handoff. Held-out listening
# regressed: the owner reports nasal coloration, slight robotization, wind-like noise and residual
# chirp. Therefore the defect is not merely mathematical continuity at a boundary; the hybrid source
# architecture itself is rejected. Two independently learned residual identities cannot be treated as
# one source by increasingly sophisticated handoff logic.
PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
PHASE_EXCLUSIVE_HANDOFF_TRAINING_EXECUTED = False
PHASE_EXCLUSIVE_HANDOFF_RAW_SAMPLEWISE_MIX_USED = False
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_NASALITY = True
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_WIND_NOISE = True
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_SLIGHT_ROBOTIZATION = True
PHASE_EXCLUSIVE_HANDOFF_OWNER_REPORTED_RESIDUAL_CHIRP = True
FURTHER_SOURCE_HANDOFF_OR_BRIDGE_TUNING_AUTHORIZED = False
HYBRID_V2_PLUS_PITCH_SYNC_PRODUCT_PATH_CLOSED = True

# Root interpretation supported by the sequence of listening gates:
# - exact Step3f residual + fixed renderer is clean;
# - fixed-frame V2 can preserve pronunciation/level but is robotic;
# - pitch-relative cycles recover near-reference voiced identity;
# - every architecture that switches from pitch-sync to an independently generated V2 residual at
#   voiced offsets leaves or worsens the terminal artifact, even when the join is mathematically C1;
# therefore the product source must be one jointly trained residual generator whose periodic and
# aperiodic behavior are two coordinates of the SAME state/output, not two checkpoint identities.
ACTIVE_ROOT_CAUSE_HYPOTHESIS = "independent_residual_source_identity_handoff_between_periodic_and_aperiodic_regions"
ACTIVE_ROOT_FIX = "single_joint_source_with_explicit_f0_phase_coordinate_and_complementary_periodic_aperiodic_energy"
ACTIVE_ROOT_FIX_REQUIRES_RETRAINING = True
ACTIVE_ROOT_FIX_POSTHOC_ENHANCEMENT = False

# Unified-source contract: one acoustic encoder, one recurrent state, and one jointly optimized
# residual waveform. A phase-harmonic component is evaluated from accumulated F0 phase and an
# aperiodic frame component is overlap-added from the same hidden state. sqrt(periodicity) and
# sqrt(1-periodicity) form a complementary energy partition inside this one source; there is no
# source switch, gate to another checkpoint, bridge, crossfade or external fallback.
UNIFIED_SOURCE_SINGLE_MODEL = True
UNIFIED_SOURCE_SINGLE_RECURRENT_STATE = True
UNIFIED_SOURCE_EXPLICIT_F0_PHASE = True
UNIFIED_SOURCE_PERIODIC_APERIODIC_COMPLEMENTARY_ENERGY = True
UNIFIED_SOURCE_SECOND_CHECKPOINT_FALLBACK = False
UNIFIED_SOURCE_HANDOFF_OR_BRIDGE = False

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

NEXT_ACTION = "implement_and_train_unified_phase_residual_source_then_compare_complete_heldout_audio_to_pitch_sync_and_identity_ceiling"
