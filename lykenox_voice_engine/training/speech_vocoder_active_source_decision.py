"""Authoritative active source-path decision for the LYKENOX vocoder.

Historical codebook and continuous-source implementations remain evidence. This file states the
active engineering path after the 2026-09-03 pitch-synchronous and phase-continuous listening gates.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v8-phase-exclusive-handoff"

ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_pitch_synchronous_residual_cycle_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_pitch_synchronous_residual_cycle_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_pitch_synchronous_cycle_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_pitch_synchronous_cycle_source_v3_phase_exclusive_handoff.py"
ACTIVE_SOURCE_DECODER = "periodic_bandlimited_fourier_with_within_cycle_next_shape_morph"
ACTIVE_SOURCE_HANDOFF = "phase_exclusive_authority_with_period_derived_c1_hermite_bridge"
ACTIVE_SOURCE_CHECKPOINT = "models/lykenox_identity/training/pitch_synchronous_cycle_source_v1/best.pt"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
SAMPLE_RATE_AUTOREGRESSIVE_MODEL = False

CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V1_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.098, 0.128, 0.153)
CONTINUOUS_SOURCE_V1_CHECKPOINT_MAY_BE_USED_FOR_PRODUCT = False

# V2 fixed amplitude collapse and produced clean/intelligible speech without gangoso, but its held-out
# voice remained audibly robotic. It remains the owned fallback only where complete pitch-sync cycles
# do not exist; it is not an accepted final product source.
CONTINUOUS_SOURCE_V2_STATUS = "positive_fallback_rejected_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
CONTINUOUS_SOURCE_V2_SPEECH_0024_PRONUNCIATION_GOOD = True
CONTINUOUS_SOURCE_V2_SPEECH_0024_GANGOSO_PRESENT = False
CONTINUOUS_SOURCE_V2_SPEECH_0024_FINAL_CHILLIDO_PRESENT = False
CONTINUOUS_SOURCE_V2_SPEECH_0024_ROBOTIC_TIMBRE_PRESENT = True
CONTINUOUS_SOURCE_V2_IDENTITY_ROUNDTRIP_CEILING_CLEAN_AND_REFERENCE_LIKE = True
CONTINUOUS_SOURCE_V2_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

# Stochastic innovation was a net regression: robotization decreased only slightly while gangoso
# returned and output level fell. No further noise/seed/mix tuning is authorized.
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
COHERENT_INNOVATION_UPDATES = 600
COHERENT_INNOVATION_BEST_VAL_TOTAL = 2.5877118905385337
COHERENT_INNOVATION_ROBOTIZATION_CHANGE = "slightly_reduced"
COHERENT_INNOVATION_GANGOSO_CHANGE = "reintroduced"
COHERENT_INNOVATION_LEVEL_CHANGE = "decreased"
COHERENT_INNOVATION_NET_PROGRESS_POSITIVE = False
COHERENT_INNOVATION_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False
FURTHER_COHERENT_INNOVATION_TUNING_AUTHORIZED = False

# Pitch-synchronous real residual cycles are the strongest learned source so far. On speech_0021 the
# owner reported V1 as almost like the original except for a robotic whistle/chirp, while V2 remained
# clean/intelligible but robotic and the identity roundtrip remained reference-like.
PITCH_SYNCHRONOUS_CYCLE_STATUS = "best_positive_learned_source_remaining_transition_chirp"
PITCH_SYNCHRONOUS_CYCLE_UPDATES = 600
PITCH_SYNCHRONOUS_CYCLE_BEST_VAL_TOTAL = 2.6395082473754883
PITCH_SYNCHRONOUS_CYCLE_SPEECH_0021_NEAR_REFERENCE_VOICE = True
PITCH_SYNCHRONOUS_CYCLE_NET_PROGRESS_POSITIVE = True
PITCH_SYNCHRONOUS_CYCLE_CODEBOOK_USED = False
PITCH_SYNCHRONOUS_CYCLE_TEACHER_FORCING_USED = False
PITCH_SYNCHRONOUS_CYCLE_THIRD_PARTY_MODEL_OR_WEIGHT_USED = False
PITCH_SYNCHRONOUS_CYCLE_REMOTE_SERVICE_USED = False
PITCH_SYNCHRONOUS_CYCLE_POSTHOC_GAIN_NORMALIZATION_USED = False
PITCH_SYNCHRONOUS_CYCLE_POSTHOC_EQ_USED = False
PITCH_SYNCHRONOUS_CYCLE_POSTHOC_DENOISING_USED = False
PITCH_SYNCHRONOUS_CYCLE_METRICS_ACCEPT_PRODUCT_QUALITY = False
PITCH_SYNCHRONOUS_CYCLE_CHECKPOINT_RETAINED = True

# V1's independent linear cycle decode had a real seam/resampling defect and is permanently rejected.
PITCH_SYNCHRONOUS_V1_HARD_SPLICE_DECODER_REJECTED = True
FURTHER_HARD_SPLICE_DECODER_USE_AUTHORIZED = False

# The periodic/band-limited phase-continuous decoder removed the hard cycle splice without training,
# but the owner still heard a low/slightly robotic voice with a whistle/chirp at the end of each word.
# Therefore cycle-edge discontinuity was not the complete root cause. The decoder is retained because
# it removes a real synthesis defect, but no further Fourier/resampling tuning is authorized.
PHASE_CONTINUOUS_DECODER_STATUS = "retained_but_insufficient_to_remove_terminal_chirp"
PHASE_CONTINUOUS_DECODER_TRAINING_EXECUTED = False
PHASE_CONTINUOUS_DECODER_PERIODIC_FOURIER = True
PHASE_CONTINUOUS_DECODER_PHYSICAL_PERIOD_BANDLIMIT = True
PHASE_CONTINUOUS_DECODER_NEXT_CYCLE_MORPH = True
PHASE_CONTINUOUS_DECODER_GLOBAL_GAIN_NORMALIZATION = False
PHASE_CONTINUOUS_DECODER_EQ = False
PHASE_CONTINUOUS_DECODER_DENOISE = False
FURTHER_PHASE_CONTINUOUS_DECODER_TUNING_AUTHORIZED = False

# New localization from listening + source-path inspection:
# - the remaining chirp is strongest at word/phrase endings, not throughout stable voiced cycles;
# - the V2 phase-continuous renderer still computes raw residual = V2*(1-gate)+pitch_sync*gate;
# - gate follows coverage*voiced*periodicity, so voiced offsets explicitly mix two independent source
#   phases while the gate falls;
# - simultaneous unrelated residual phases can beat and create a terminal tone/timbre thickening.
# The correction is source authority, not another filter: pitch-sync alone inside complete cycles,
# V2 alone outside, with only a short C1 value/slope bridge at each authority boundary.
ACTIVE_ROOT_CAUSE_HYPOTHESIS = "raw_samplewise_mix_of_phase_independent_sources_during_voiced_handoff"
ACTIVE_ROOT_FIX = "exclusive_source_authority_with_period_derived_c1_transition_bridge"
ACTIVE_ROOT_FIX_REQUIRES_RETRAINING = False
ACTIVE_ROOT_FIX_REUSES_EXISTING_BEST_CHECKPOINT = True
ACTIVE_ROOT_FIX_POSTHOC_ENHANCEMENT = False
RAW_V2_PITCH_SYNC_SAMPLEWISE_CROSSFADE_AUTHORIZED = False
PHASE_EXCLUSIVE_SOURCE_AUTHORITY = True
C1_HANDOFF_VALUE_CONTINUITY = True
C1_HANDOFF_SLOPE_CONTINUITY = True
C1_HANDOFF_WIDTH_DERIVED_FROM_LOCAL_F0_PERIOD = True

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
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

NEXT_ACTION = "rerender_existing_pitch_sync_checkpoint_with_phase_exclusive_c1_handoff_then_listen_for_terminal_chirp_removal"
