"""Focused source-path decision after residual-statistics V1 failed human listening.

Policy: LYX-POL-001. This file intentionally authorizes no training or renderer changes.
The only active gate is the phase-vs-magnitude hybrid listening forensic on the two known-good
real-residual oracle utterances.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-source-decision-v20-phase-magnitude-forensic"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
REAL_RESIDUAL_ORACLE_RETAINED = True

GOLD_ORACLE_SPEECH_0021 = "speech_0021_6cd35984e877_seg_001__identity_roundtrip_ceiling.wav"
GOLD_ORACLE_SPEECH_0022 = "speech_0022_ba721f6129b9_seg_005__real_residual_resynthesis.wav"

RESIDUAL_STATISTICS_SOURCE_V1_STATUS = "rejected_human_listening_mal_sintonizado"
RESIDUAL_STATISTICS_SOURCE_V1_STRUCTURAL_GRID_GATE_HAD_PASSED = True
RESIDUAL_STATISTICS_SOURCE_V1_METRICS_ACCEPT_PRODUCT_QUALITY = False
FURTHER_RESIDUAL_STATISTICS_SOURCE_V1_TRAINING_AUTHORIZED = False

# New controlled forensic: no training, no optimizer, no renderer changes, no post-processing.
PHASE_MAGNITUDE_FORENSIC_STATUS = "awaiting_hybrid_human_listening"
PHASE_MAGNITUDE_FORENSIC_SCRIPT = "scripts/diagnostic_residual_phase_magnitude_forensic_v1.py"
TRAINING_EXECUTED = False
OPTIMIZER_CREATED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
FURTHER_SOURCE_TRAINING_AUTHORIZED = False

SPEECH_0021_PHASE_ALIGNMENT_TARGET_VS_CANDIDATE = -0.000497
SPEECH_0021_LOG_MAGNITUDE_L1_TARGET_VS_CANDIDATE = 0.9904
SPEECH_0022_PHASE_ALIGNMENT_TARGET_VS_CANDIDATE = -0.003934
SPEECH_0022_LOG_MAGNITUDE_L1_TARGET_VS_CANDIDATE = 1.3186

# The objective values establish that the generated residual differs strongly from the target in both
# phase and log magnitude. They do not by themselves identify which difference causes the audible
# tuning/radio defect. Only the controlled hybrids can localize that perceptual failure.
OBJECTIVE_METRICS_LOCALIZE_ROOT_CAUSE_ALONE = False

HYBRID_A = "target_magnitude_plus_candidate_phase"
HYBRID_B = "candidate_magnitude_plus_target_phase"

INTERPRETATION_A_BAD_B_GOOD = "candidate_phase_or_temporal_coherence_is_primary_failure"
INTERPRETATION_A_GOOD_B_BAD = "candidate_magnitude_or_spectral_microstructure_is_primary_failure"
INTERPRETATION_A_BAD_B_BAD = "joint_phase_magnitude_structure_is_required_and_independent_factorization_is_invalid"
INTERPRETATION_A_GOOD_B_GOOD = "full_candidate_residual_construction_coupling_is_primary_failure"

ACTIVE_ENGINEERING_GATE = "listen_only_to_phase_magnitude_hybrids_against_known_good_real_residual_oracle"
NEXT_ACTION = "classify_hybrid_A_and_hybrid_B_on_speech_0021_and_speech_0022_before_any_new_training_or_architecture"
