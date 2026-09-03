"""Authoritative active engineering decision for the LYKENOX vocoder source path.

This file records the direct-reference, common-conditioning, controlled conditioning-V2 retrain and
stable-voiced residual-forensics gate reached on 2026-09-03. Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v15-stable-voiced-residual-coherence-forensics"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

# Historical source evidence.
CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V2_STATUS = "positive_historical_baseline_rejected_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
PITCH_SYNCHRONOUS_CYCLE_STATUS = "strongest_voiced_representation_not_complete_source"
PITCH_SYNCHRONOUS_CYCLE_SPEECH_0021_NEAR_REFERENCE_VOICE = True
PHASE_EXCLUSIVE_HANDOFF_STATUS = "rejected_for_nasal_wind_robotic_regression"
UNIFIED_SOURCE_RUN_STATUS = "rejected_same_audible_defects_persist"

# Direct generated-vs-reference evidence. Real residual / identity roundtrip remain clean and closest
# to reference. Learned sources share tonal excess and upper-band deficits.
CROSS_VARIANT_REFERENCE_DIAGNOSTIC_STATUS = "complete"
DIRECT_REFERENCE_REAL_RESIDUAL_SCORE = 3.4813
DIRECT_REFERENCE_REAL_RESIDUAL_RMS_DELTA_DB = 0.05
DIRECT_REFERENCE_CONTINUOUS_V2_SCORE = 7.3656
DIRECT_REFERENCE_CONTINUOUS_V2_TONAL_PROMINENCE_EXCESS_P95_DB = 7.62
SHARED_ANOMALY_SPEECH_0021_TIME_SECONDS = 5.80
SHARED_ANOMALY_SPEECH_0024_PRIMARY_TIME_SECONDS = 4.00
SHARED_ANOMALY_SPEECH_0024_PRIMARY_MEAN_TONAL_EXCESS_DB = 16.576
SHARED_ANOMALY_SPEECH_0024_SECONDARY_TIME_SECONDS = 5.20

# Pitch-conditioning V1 transition semantics were genuinely inconsistent: hard-zeroed F0 and binary
# voiced could coexist with non-zero raw periodicity. Conditioning V2 corrected that semantics, but
# the controlled retrain demonstrates only a small objective improvement rather than a strong defect
# correction. Therefore no further training is authorized from this hypothesis.
PITCH_V1_TRANSITION_SEMANTICS_REJECTED = True
PITCH_V1_OCTAVE_ERROR_IS_PRIMARY_ROOT_CAUSE = False
PITCH_CONDITIONING_V2 = "lykenox-pitch-conditioning-v2-continuous-strength"
PITCH_CONDITIONING_V2_AUDIT_STATUS = "complete_partial_root_correction_confirmed"
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_STATUS = "complete_small_objective_improvement_not_solution"
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_UPDATES = 600
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_BEST_VAL_TOTAL = 2.899257183074951
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_ARCHITECTURE_CHANGED = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_LOSS_CHANGED = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_RENDERER_CHANGED = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_STEP3F_TARGET_CHANGED = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_RMS_RATIOS = (0.8470, 0.8729, 0.9176)
PITCH_CONDITIONING_V2_BASELINE_RMS_RATIOS = (0.8331, 0.9048, 0.9493)

# Direct candidate-vs-baseline anomaly comparison after the controlled retrain.
# speech_0021 ~5.80 s: tonal excess fell 9.37 -> 7.64 dB, but air excess remained 10.95 -> 10.94 dB.
# speech_0024 ~5.20 s: tonal excess 7.64 -> 6.73 dB and air 7.65 -> 7.36 dB: only mild/moderate.
# speech_0024 ~4.00 s: tonal excess 17.08 -> 16.17 dB: still extremely high.
SPEECH_0021_5S8_BASELINE_TONAL_EXCESS_DB = 9.37
SPEECH_0021_5S8_CANDIDATE_TONAL_EXCESS_DB = 7.64
SPEECH_0021_5S8_BASELINE_AIR_EXCESS_DB = 10.95
SPEECH_0021_5S8_CANDIDATE_AIR_EXCESS_DB = 10.94
SPEECH_0024_5S2_BASELINE_TONAL_EXCESS_DB = 7.64
SPEECH_0024_5S2_CANDIDATE_TONAL_EXCESS_DB = 6.73
SPEECH_0024_5S2_BASELINE_AIR_EXCESS_DB = 7.65
SPEECH_0024_5S2_CANDIDATE_AIR_EXCESS_DB = 7.36
SPEECH_0024_4S_BASELINE_TONAL_EXCESS_DB = 17.08
SPEECH_0024_4S_CANDIDATE_TONAL_EXCESS_DB = 16.17
PITCH_CONDITIONING_V2_FURTHER_RETRAINING_AUTHORIZED = False
PITCH_CONDITIONING_V2_IS_AUDIBLE_ROOT_SOLUTION = False

# Stable voiced defect is now the primary open root problem. At speech_0024 ~4.00 s F0 is stable
# (~86.96 Hz), the frame is strongly voiced/energized, there is no confirmed octave error, and every
# learned source remains far too tonal while exact Step3f residual resynthesis is clean. Therefore the
# next gate moves before the renderer and compares the learned residual itself with the real residual.
STABLE_VOICED_TONAL_EXCESS_REMAINS_PRIMARY_OPEN_DEFECT = True
SPEECH_0024_4S_F0_STABLE = True
SPEECH_0024_4S_NEAR_OCTAVE_COMPETITOR = False
STABLE_VOICED_RESIDUAL_FORENSIC = "scripts/run_stable_voiced_residual_forensics_v1.py"
STABLE_VOICED_RESIDUAL_FORENSIC_HYPOTHESIS = "learned_residual_is_excessively_cycle_coherent_before_renderer"
STABLE_VOICED_RESIDUAL_FORENSIC_TRAINS_MODEL = False
STABLE_VOICED_RESIDUAL_FORENSIC_WRITES_WAV = False
STABLE_VOICED_RESIDUAL_FORENSIC_EXECUTES_RENDERER = False
STABLE_VOICED_RESIDUAL_FORENSIC_METRICS = (
    "f0_lag_correlation",
    "coherent_cycle_energy_fraction",
    "cycle_deviation_rms",
    "high_band_spectral_flatness",
)

# No new source architecture or source training is authorized until this residual-domain hypothesis is
# accepted or rejected from direct target-vs-prediction evidence. No post-hoc enhancement is allowed.
ACTIVE_ENGINEERING_GATE = "stable_voiced_real_vs_predicted_residual_cycle_coherence_forensics"
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False

NEXT_ACTION = "run_stable_voiced_residual_forensics_before_any_new_source_training_or_architecture_change"
