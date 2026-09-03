"""Authoritative active engineering decision for the LYKENOX vocoder source path.

This file records the direct-reference, conditioning and residual-domain evidence reached on
2026-09-03. Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v18-close-frame-waveform-regression"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

# Historical source evidence.
CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression_and_per_frame_local_innovation_geometry"
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
UNIFIED_SOURCE_RUN_STATUS = "rejected_same_audible_defects_persist"

DIRECT_REFERENCE_REAL_RESIDUAL_SCORE = 3.4813
DIRECT_REFERENCE_CONTINUOUS_V2_SCORE = 7.3656
SHARED_ANOMALY_SPEECH_0021_TIME_SECONDS = 5.80
SHARED_ANOMALY_SPEECH_0024_PRIMARY_TIME_SECONDS = 4.00
SHARED_ANOMALY_SPEECH_0024_PRIMARY_MEAN_TONAL_EXCESS_DB = 16.576
SHARED_ANOMALY_SPEECH_0024_SECONDARY_TIME_SECONDS = 5.20

# Conditioning-v2 corrected a real transition-semantic inconsistency but produced only a small
# objective improvement and is closed as an audible root solution. Its coherent continuous semantic
# contract remains usable as conditioning for future owned source models.
PITCH_V1_TRANSITION_SEMANTICS_REJECTED = True
PITCH_V1_OCTAVE_ERROR_IS_PRIMARY_ROOT_CAUSE = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_STATUS = "complete_small_objective_improvement_not_solution"
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_BEST_VAL_TOTAL = 2.899257183074951
PITCH_CONDITIONING_V2_FURTHER_RETRAINING_AUTHORIZED = False
PITCH_CONDITIONING_V2_IS_AUDIBLE_ROOT_SOLUTION = False
PITCH_CONDITIONING_V2_MAY_BE_USED_AS_COHERENT_CONDITIONING_CONTRACT = True

# Stable-voiced residual forensics reject excessive F0-cycle coherence. The learned residuals are not
# more correlated with F0 than the real residual. The dominant pre-renderer defect is collapse of
# high-band spectral flatness.
STABLE_VOICED_F0_COHERENCE_HYPOTHESIS_STATUS = "rejected"
SPEECH_0024_4S_REAL_F0_LAG_CORRELATION = 0.0342
SPEECH_0024_4S_V2_F0_LAG_CORRELATION = 0.00017
SPEECH_0024_4S_CONDITIONING_V2_F0_LAG_CORRELATION = -0.00003
SPEECH_0024_4S_REAL_HIGH_BAND_SPECTRAL_FLATNESS = 0.8435
SPEECH_0024_4S_V2_HIGH_BAND_SPECTRAL_FLATNESS = 0.0703
SPEECH_0024_4S_CONDITIONING_V2_HIGH_BAND_SPECTRAL_FLATNESS = 0.1054

# Hop-grid forensics confirm that independently predicted overlapping 512/256 vectors violate the
# duplicated-sample consistency of the exact real Step-3f analysis representation and create a nearly
# total 93.75 Hz high-band comb before the minimum-phase renderer.
HOP_GRID_FREQUENCY_HZ = 93.75
LEARNED_512_256_OVERLAP_ROOT_CAUSE_STATUS = "confirmed_structural_representation_failure"
SPEECH_0024_4S_REAL_HOP_LAG_CORRELATION = 0.0071
SPEECH_0024_4S_V2_HOP_LAG_CORRELATION = 0.9994
SPEECH_0024_4S_REAL_OVERLAP_RELATIVE_DISAGREEMENT_RMS = 0.000000032
SPEECH_0024_4S_V2_OVERLAP_RELATIVE_DISAGREEMENT_RMS = 1.3124
SPEECH_0024_4S_REAL_HIGH_BAND_GRID_HARMONIC_POWER_FRACTION = 0.1766
SPEECH_0024_4S_V2_HIGH_BAND_GRID_HARMONIC_POWER_FRACTION = 0.9996
REAL_512_256_ANALYSIS_SYNTHESIS_REPRESENTATION_REMAINS_VALID_FOR_EXACT_TARGET_ROUNDTRIP = True
LEARNED_512_256_OVERLAPPING_VECTOR_SOURCE_REPRESENTATION_REJECTED = True
FURTHER_LEARNED_512_256_SOURCE_TRAINING_AUTHORIZED = False
FURTHER_OVERLAP_CONSISTENCY_LOSS_PATCHING_AUTHORIZED = False

# The non-overlapping 256-sample stream source removed duplicated sample authority but failed its first
# structural gate. Therefore overlap inconsistency was one real failure mechanism but not the deepest
# cause of the 93.75 Hz comb. With no overlap at all, the deterministic frame waveform head still
# converged to an almost repeated 256-sample template.
CONTINUOUS_STREAM_SOURCE_STATUS = "rejected_first_structural_gate_frame_template_repetition_persists"
CONTINUOUS_STREAM_SOURCE_UPDATES = 600
CONTINUOUS_STREAM_SOURCE_BEST_VAL_TOTAL = 2.8932310740152993
CONTINUOUS_STREAM_SOURCE_REPRESENTATION = "unique_contiguous_256_sample_blocks_no_overlap"
CONTINUOUS_STREAM_SOURCE_OVERLAP_SAMPLES = 0
CONTINUOUS_STREAM_SOURCE_DUPLICATED_SAMPLE_AUTHORITY = False
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_TARGET_HOP_LAG_CORRELATION = 0.0015
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_CANDIDATE_HOP_LAG_CORRELATION = 0.9975
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_V2_HOP_LAG_CORRELATION = 0.9995
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_TARGET_HIGH_BAND_FLATNESS = 0.8440
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_CANDIDATE_HIGH_BAND_FLATNESS = 0.2572
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_V2_HIGH_BAND_FLATNESS = 0.1465
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_TARGET_GRID_POWER_FRACTION = 0.1303
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_CANDIDATE_GRID_POWER_FRACTION = 0.9931
CONTINUOUS_STREAM_SOURCE_SPEECH_0024_V2_GRID_POWER_FRACTION = 0.9967
CONTINUOUS_STREAM_SOURCE_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False
FURTHER_CONTINUOUS_STREAM_WAVEFORM_TRAINING_AUTHORIZED = False

# Root interpretation after the stream failure: deterministic point regression of residual waveform
# phase at one acoustic-frame output block is itself structurally invalid. Acoustic conditioning can
# determine residual level, spectral envelope and periodicity, but not the exact aperiodic fine-phase
# realization. A deterministic sample head therefore collapses toward a stable frame-aligned template,
# recreating the hop-grid comb even without overlap.
DETERMINISTIC_FRAME_WAVEFORM_REGRESSION_STATUS = "closed_structural_failure"
DETERMINISTIC_FRAME_WAVEFORM_HEAD_256_AUTHORIZED = False
DETERMINISTIC_FRAME_WAVEFORM_HEAD_512_AUTHORIZED = False
WAVEFORM_SAMPLE_L1_AS_PRIMARY_SOURCE_TARGET_AUTHORIZED = False
WAVEFORM_BLOCK_COSINE_AS_PRIMARY_SOURCE_TARGET_AUTHORIZED = False

# Old coherent-innovation does not reopen this route: it generated local pseudo-noise separately per
# 512-sample vector and mixed it into a still-deterministic waveform-vector path, so phase/noise still
# reset on the frame/vector geometry and the model retained the rejected sample-regression surface.
OLD_COHERENT_INNOVATION_REOPEN_AUTHORIZED = False

# Active root correction: predict only identifiable residual statistics at frame rate. No neural head
# emits waveform samples. Predicted source cepstrum, explicit log-RMS and residual periodicity drive a
# single continuous full-utterance source carrier. F0 phase and deterministic owned noise use absolute
# sample position and never reset per frame. The source carrier is shaped by a time-varying
# minimum-phase source filter; the fixed vocal-tract renderer remains unchanged.
ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_residual_statistics_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_minimum_phase_residual_statistics_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_residual_statistics_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_residual_statistics_source_v1.py"
ACTIVE_SOURCE_ENTRYPOINT = "scripts/train_residual_statistics_source_v1.py"
ACTIVE_SOURCE_RUN_DIR = "models/lykenox_identity/training/residual_statistics_source_v1"
ACTIVE_SOURCE_REPRESENTATION = "frame_source_statistics_plus_continuous_absolute_phase_carrier"
ACTIVE_SOURCE_PREDICTS_WAVEFORM_SAMPLES = False
ACTIVE_SOURCE_PREDICTS_SOURCE_CEPSTRUM = True
ACTIVE_SOURCE_PREDICTS_EXPLICIT_LOG_RMS = True
ACTIVE_SOURCE_PREDICTS_RESIDUAL_PERIODICITY = True
ACTIVE_SOURCE_CARRIER_PHASE_RESETS_PER_FRAME = False
ACTIVE_SOURCE_CARRIER_NOISE_RESETS_PER_FRAME = False
ACTIVE_SOURCE_CONDITIONING_CONTRACT = "lykenox-pitch-conditioning-v2-continuous-strength"
ACTIVE_SOURCE_STATUS = "root_correction_implemented_training_authorized"

ACTIVE_ENGINEERING_GATE = "train_statistics_source_then_reject_before_listening_unless_93_75hz_grid_signature_collapses"
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = True
AUTHORIZED_TRAINING_ARCHITECTURE = ACTIVE_SOURCE_ARCHITECTURE
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False

NEXT_ACTION = "train_residual_statistics_source_v1_then_verify_hop_lag_flatness_and_grid_power_before_any_human_acceptance"
