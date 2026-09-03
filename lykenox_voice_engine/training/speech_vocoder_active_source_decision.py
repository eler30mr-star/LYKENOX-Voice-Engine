"""Authoritative active engineering decision for the LYKENOX vocoder source path.

This file records the direct-reference, conditioning and residual-domain evidence reached on
2026-09-03. Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v17-hop-grid-root-confirmed-stream-source"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

# Historical source evidence.
CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
UNIFIED_SOURCE_RUN_STATUS = "rejected_same_audible_defects_persist"

DIRECT_REFERENCE_REAL_RESIDUAL_SCORE = 3.4813
DIRECT_REFERENCE_CONTINUOUS_V2_SCORE = 7.3656
SHARED_ANOMALY_SPEECH_0021_TIME_SECONDS = 5.80
SHARED_ANOMALY_SPEECH_0024_PRIMARY_TIME_SECONDS = 4.00
SHARED_ANOMALY_SPEECH_0024_PRIMARY_MEAN_TONAL_EXCESS_DB = 16.576
SHARED_ANOMALY_SPEECH_0024_SECONDARY_TIME_SECONDS = 5.20

# Conditioning V2 corrected a real transition-semantic inconsistency but produced only a small
# objective improvement and is closed as an audible root solution. Its coherent continuous semantic
# contract may still be used by future source models.
PITCH_V1_TRANSITION_SEMANTICS_REJECTED = True
PITCH_V1_OCTAVE_ERROR_IS_PRIMARY_ROOT_CAUSE = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_STATUS = "complete_small_objective_improvement_not_solution"
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_UPDATES = 600
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_BEST_VAL_TOTAL = 2.899257183074951
PITCH_CONDITIONING_V2_FURTHER_RETRAINING_AUTHORIZED = False
PITCH_CONDITIONING_V2_IS_AUDIBLE_ROOT_SOLUTION = False
PITCH_CONDITIONING_V2_MAY_BE_USED_AS_COHERENT_CONDITIONING_CONTRACT = True

# Stable-voiced forensics rejected excessive F0-cycle coherence as the tonal root cause. Learned
# residuals are actually less correlated with F0 than the clean real Step-3f residual. The dominant
# discrepancy before the renderer is collapse of high-band spectral flatness.
STABLE_VOICED_F0_COHERENCE_HYPOTHESIS_STATUS = "rejected"
SPEECH_0024_4S_REAL_F0_LAG_CORRELATION = 0.0342
SPEECH_0024_4S_V2_F0_LAG_CORRELATION = 0.00017
SPEECH_0024_4S_CONDITIONING_V2_F0_LAG_CORRELATION = -0.00003
SPEECH_0024_4S_REAL_HIGH_BAND_SPECTRAL_FLATNESS = 0.8435
SPEECH_0024_4S_V2_HIGH_BAND_SPECTRAL_FLATNESS = 0.0703
SPEECH_0024_4S_CONDITIONING_V2_HIGH_BAND_SPECTRAL_FLATNESS = 0.1054

# Hop-grid forensics confirm the root representation defect at speech_0024 ~4.00 s. The real Step-3f
# 512/256 analysis vectors are mutually consistent because duplicated overlap samples derive from one
# underlying continuous residual. Learned V2 vectors are not constrained to represent the duplicated
# samples consistently. Their mismatch repeats every 256 samples (24 kHz/256 = 93.75 Hz) and produces
# a near-total high-band comb on hop-grid harmonics before the minimum-phase renderer.
HOP_GRID_FREQUENCY_HZ = 93.75
HOP_GRID_HYPOTHESIS = "predicted_512_256_vectors_break_overlap_consistency_and_create_hop_grid_comb_tonality"
HOP_GRID_HYPOTHESIS_CONFIRMED = True
HOP_GRID_ROOT_CAUSE_STATUS = "confirmed_structural_representation_failure"
SPEECH_0024_4S_REAL_HOP_LAG_CORRELATION = 0.0071
SPEECH_0024_4S_V2_HOP_LAG_CORRELATION = 0.9994
SPEECH_0024_4S_CONDITIONING_V2_HOP_LAG_CORRELATION = 0.9987
SPEECH_0024_4S_REAL_OVERLAP_RELATIVE_DISAGREEMENT_RMS = 0.000000032
SPEECH_0024_4S_V2_OVERLAP_RELATIVE_DISAGREEMENT_RMS = 1.3124
SPEECH_0024_4S_CONDITIONING_V2_OVERLAP_RELATIVE_DISAGREEMENT_RMS = 1.0074
SPEECH_0024_4S_REAL_OVERLAP_ADJACENT_ESTIMATE_CORRELATION = 1.0
SPEECH_0024_4S_V2_OVERLAP_ADJACENT_ESTIMATE_CORRELATION = 0.3992
SPEECH_0024_4S_CONDITIONING_V2_OVERLAP_ADJACENT_ESTIMATE_CORRELATION = 0.6034
SPEECH_0024_4S_REAL_HIGH_BAND_GRID_HARMONIC_POWER_FRACTION = 0.1766
SPEECH_0024_4S_V2_HIGH_BAND_GRID_HARMONIC_POWER_FRACTION = 0.9996
SPEECH_0024_4S_CONDITIONING_V2_HIGH_BAND_GRID_HARMONIC_POWER_FRACTION = 0.9988

# Important scope: overlap-add itself remains mathematically valid for real/consistent vectors. What
# is rejected is using independently predicted overlapping 512-sample vectors as the learned source
# representation, because two network outputs are then allowed to assign incompatible values to the
# same underlying 256 samples.
REAL_512_256_ANALYSIS_SYNTHESIS_REPRESENTATION_REMAINS_VALID_FOR_EXACT_TARGET_ROUNDTRIP = True
LEARNED_512_256_OVERLAPPING_VECTOR_SOURCE_REPRESENTATION_REJECTED = True
FURTHER_LEARNED_512_256_SOURCE_TRAINING_AUTHORIZED = False
FURTHER_OVERLAP_CONSISTENCY_LOSS_PATCHING_AUTHORIZED = False

# Root correction: one frame owns one contiguous 256-sample residual block. Blocks concatenate
# directly into the residual stream; there is no source OLA, duplicated sample authority, previous
# waveform feedback or teacher forcing. The fixed renderer remains byte-for-byte unchanged. Loss
# authority intentionally matches Continuous Source V2 while the representation geometry changes.
ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_continuous_residual_stream_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_minimum_phase_continuous_stream_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_continuous_stream_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_continuous_residual_stream_source_v1.py"
ACTIVE_SOURCE_ENTRYPOINT = "scripts/train_continuous_residual_stream_source_v1.py"
ACTIVE_SOURCE_RUN_DIR = "models/lykenox_identity/training/continuous_residual_stream_source_v1"
ACTIVE_SOURCE_RESIDUAL_REPRESENTATION = "unique_contiguous_256_sample_blocks_no_overlap"
ACTIVE_SOURCE_OVERLAP_SAMPLES = 0
ACTIVE_SOURCE_DUPLICATED_SAMPLE_AUTHORITY = False
ACTIVE_SOURCE_PREVIOUS_WAVEFORM_FEEDBACK = False
ACTIVE_SOURCE_TEACHER_FORCING = False
ACTIVE_SOURCE_CONDITIONING_CONTRACT = "lykenox-pitch-conditioning-v2-continuous-strength"
ACTIVE_SOURCE_LOSS_WEIGHTS_MATCH_CONTINUOUS_V2 = True
ACTIVE_SOURCE_STATUS = "root_correction_implemented_training_authorized"

ACTIVE_ENGINEERING_GATE = "train_non_overlapping_unique_sample_stream_source_then_verify_hop_grid_signature_and_full_heldout_audio"
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = True
AUTHORIZED_TRAINING_ARCHITECTURE = ACTIVE_SOURCE_ARCHITECTURE
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False

NEXT_ACTION = "train_continuous_residual_stream_source_v1_and_reject_unless_93_75hz_grid_signature_collapses_before_renderer_and_human_heldout_quality_improves"
