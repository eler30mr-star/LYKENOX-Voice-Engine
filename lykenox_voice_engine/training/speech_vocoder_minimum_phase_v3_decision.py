"""Current minimum-phase vocoder decision after the failed v2 authority preflight.

This file supersedes the open-gate state recorded in the earlier architecture contract without
rewriting historical evidence.  The v2 equal-norm weight candidate is rejected for training
because real owned probes showed negative common-direction alignments/descent dots.  The only
active path is deterministic pre-training directional calibration -> disjoint verification ->
one fixed run weight vector -> bounded CPU training -> complete held-out listening.
"""

DECISION_VERSION = "owned-minimum-phase-v3-directional-decision-v1"
POLICY_ID = "LYX-POL-001"
SUPERSEDES_GATE_STATE_FROM = "owned-vocoder-architecture-contract-v1"
ARCHITECTURE_FAMILY = "owned_minimum_phase_time_varying_filter_over_neutral_excitation"

V2_WEIGHT_CONTRACT_VERSION = "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2"
V2_WEIGHT_CONTRACT_STATUS = "rejected_for_training_directional_conflict"
V2_PREFLIGHT_STATUS = "fail"
V2_PREFLIGHT_OPTIMIZER_CREATED = False
V2_PREFLIGHT_PARAMETER_UPDATE_EXECUTED = False
V2_PREFLIGHT_CHECKPOINT_SAVED = False
V2_FAILURE_EVIDENCE = {
    "cepstrum_neutral_spectral_balance_alignment": -0.0285,
    "cepstrum_neutral_spectral_balance_descent_dot": -591.84,
    "cepstrum_connected_presence_alignment": -0.0108,
    "cepstrum_connected_spectral_balance_alignment": -0.0134,
    "cepstrum_connected_presence_descent_dot": -131647.89,
    "cepstrum_connected_spectral_balance_descent_dot": -51244.09,
    "parameter_neutral_envelope_alignment": -0.0519,
    "parameter_neutral_envelope_descent_dot": -1580.42,
    "parameter_connected_presence_alignment": -0.0216,
    "parameter_connected_presence_descent_dot": -1879.15,
}

ACTIVE_OBJECTIVE_VERSION = "owned-minimum-phase-objective-v3-directional-fixed"
ACTIVE_CALIBRATION_VERSION = "owned-minimum-phase-directional-fixed-weight-calibration-v1"
ACTIVE_TRAINER_VERSION = "owned-minimum-phase-resumable-trainer-v3-directional-fixed"
ACTIVE_PIPELINE_VERSION = "owned-minimum-phase-train-and-listen-v3-directional-fixed"
CPU_ONLY = True
DIRECTIONAL_CALIBRATION_BEFORE_MODEL_OPTIMIZER_REQUIRED = True
DISJOINT_DIRECTIONAL_VERIFICATION_REQUIRED = True
WEIGHTS_FIXED_FOR_WHOLE_RUN_AND_RESUME = True
ADAPTIVE_REWEIGHTING_DURING_TRAINING_AUTHORIZED = False
RUNTIME_REDERIVATION_AFTER_TRAINING_START_AUTHORIZED = False
BOUNDED_MODEL_OPTIMIZER_AUTHORIZED_AFTER_VERIFIED_CALIBRATION_ONLY = True
MAX_UPDATES_AUTHORIZED = 400
SCOPED_NEW_CHECKPOINT_AUTHORIZED_AFTER_VERIFIED_CALIBRATION_ONLY = True
GENERAL_PERSISTENT_TRAINING_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELD_OUT_AUDIO_REQUIRED_FOR_PRODUCT_ACCEPTANCE = True

NEXT_ACTION = "run_integrated_directional_calibration_then_bounded_train_and_listen"
