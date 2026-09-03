"""Authoritative active engineering decision for the LYKENOX vocoder source path.

This file records the direct-reference, conditioning and residual-domain evidence reached on
2026-09-03. Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v16-hop-grid-overlap-forensics"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

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

PITCH_V1_TRANSITION_SEMANTICS_REJECTED = True
PITCH_V1_OCTAVE_ERROR_IS_PRIMARY_ROOT_CAUSE = False
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_STATUS = "complete_small_objective_improvement_not_solution"
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_UPDATES = 600
PITCH_CONDITIONING_V2_CONTROLLED_RETRAIN_BEST_VAL_TOTAL = 2.899257183074951
PITCH_CONDITIONING_V2_FURTHER_RETRAINING_AUTHORIZED = False
PITCH_CONDITIONING_V2_IS_AUDIBLE_ROOT_SOLUTION = False

# Stable voiced residual forensics at speech_0024 ~4.00 s rejects the prior F0-cycle coherence
# hypothesis. The learned residuals are NOT more correlated/coherent at the cached F0 than the real
# Step-3f residual. Instead, the dominant discrepancy is a catastrophic collapse in high-band
# spectral flatness before the renderer.
STABLE_VOICED_F0_COHERENCE_HYPOTHESIS_STATUS = "rejected"
SPEECH_0024_4S_REAL_F0_LAG_CORRELATION = 0.0342
SPEECH_0024_4S_V2_F0_LAG_CORRELATION = 0.00017
SPEECH_0024_4S_CONDITIONING_V2_F0_LAG_CORRELATION = -0.00003
SPEECH_0024_4S_REAL_COHERENT_CYCLE_ENERGY_FRACTION = 0.0216
SPEECH_0024_4S_V2_COHERENT_CYCLE_ENERGY_FRACTION = 0.0136
SPEECH_0024_4S_CONDITIONING_V2_COHERENT_CYCLE_ENERGY_FRACTION = 0.0094
SPEECH_0024_4S_REAL_HIGH_BAND_SPECTRAL_FLATNESS = 0.8435
SPEECH_0024_4S_V2_HIGH_BAND_SPECTRAL_FLATNESS = 0.0703
SPEECH_0024_4S_CONDITIONING_V2_HIGH_BAND_SPECTRAL_FLATNESS = 0.1054
SPEECH_0024_4S_CONDITIONING_V2_MINUS_REAL_HIGH_BAND_FLATNESS = -0.7380

# The 512/256 representation is exact for target vectors, so overlap-add itself is not rejected.
# The active hypothesis is narrower: independently predicted overlapping vectors may disagree about
# the same duplicated 256 underlying samples. That disagreement repeats every 256 samples, i.e.
# 24 kHz / 256 = 93.75 Hz, and may create a frame-grid comb/tonal texture while remaining unrelated
# to F0. The dedicated forensic measures de-windowed overlap disagreement, hop-lag correlation and
# high-band energy concentrated near harmonics of 93.75 Hz before the minimum-phase renderer.
HOP_GRID_FORENSIC = "scripts/run_hop_grid_residual_forensics_v1.py"
HOP_GRID_FREQUENCY_HZ = 93.75
HOP_GRID_HYPOTHESIS = "predicted_512_256_vectors_break_overlap_consistency_and_create_hop_grid_comb_tonality"
HOP_GRID_HYPOTHESIS_CONFIRMED = False
HOP_GRID_FORENSIC_TRAINS_MODEL = False
HOP_GRID_FORENSIC_WRITES_WAV = False
HOP_GRID_FORENSIC_EXECUTES_RENDERER = False
HOP_GRID_FORENSIC_WRITES_CHECKPOINT = False
HOP_GRID_FORENSIC_METRICS = (
    "hop_lag_correlation",
    "overlap_relative_disagreement_rms",
    "overlap_adjacent_estimate_correlation_mean",
    "high_band_grid_harmonic_power_fraction",
)

ACTIVE_ENGINEERING_GATE = "test_predicted_overlap_consistency_and_93_75hz_grid_comb_before_any_new_training"
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False

NEXT_ACTION = "run_hop_grid_residual_forensics_before_any_new_source_training_or_architecture_change"
