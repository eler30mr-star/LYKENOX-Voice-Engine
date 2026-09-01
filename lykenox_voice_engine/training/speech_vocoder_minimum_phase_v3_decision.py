"""Current minimum-phase vocoder decision after directional and real-residual diagnostics.

This file supersedes the open-gate state recorded in the earlier architecture contract without
rewriting historical evidence. The v2 equal-norm weight candidate remains rejected because real
owned probes showed negative common-direction alignments/descent dots.

Pure-DSP oracle diagnostics isolated the dominant audible degradation. Removing the synthetic
pulse+noise excitation and resynthesizing with the owned real residual produced a clean, natural
result reported by the owner as sounding like the original voice audio. Therefore bounded model
training remains blocked while the source excitation is redesigned and validated independently.

A new owned candidate is now implemented but NOT integrated into production: deterministic
Rosenberg glottal pulses calibrated from owned real-residual cycles plus measured multi-band
aperiodicity. It cannot be evaluated until its two local owned calibration artifacts are generated,
and it cannot become active until complete held-out oracle listening demonstrates a clear benefit.
"""

DECISION_VERSION = "owned-minimum-phase-v3-decision-v3-calibrated-glottal-candidate"
POLICY_ID = "LYX-POL-001"
SUPERSEDES_GATE_STATE_FROM = "owned-vocoder-architecture-contract-v1"
ARCHITECTURE_FAMILY = "owned_minimum_phase_time_varying_filter_over_calibrated_excitation_candidate"

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
MAX_UPDATES_AUTHORIZED = 400
GENERAL_PERSISTENT_TRAINING_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELD_OUT_AUDIO_REQUIRED_FOR_PRODUCT_ACCEPTANCE = True

# Pure-DSP renderer/excitation isolation evidence.
REAL_RESIDUAL_DIAGNOSTIC_SCRIPT = "scripts/diagnostic_real_residual_resynthesis_v1.py"
REAL_RESIDUAL_EVIDENCE_DOC = "docs/LYKENOX_VOCODER_MINIMUM_PHASE_REAL_RESIDUAL_EVIDENCE.md"
REAL_RESIDUAL_MODEL_USED = False
REAL_RESIDUAL_TRAINING_EXECUTED = False
REAL_RESIDUAL_CHECKPOINT_USED = False
REAL_RESIDUAL_SYNTHETIC_EXCITATION_USED = False
REAL_RESIDUAL_HUMAN_LISTENING_RESULT = "clean_natural_matches_original_voice_audio"
REAL_RESIDUAL_FILTER_ENVELOPE_PATH_STATUS = "clean_oracle_resynthesis_demonstrated"
SYNTHETIC_EXCITATION_STATUS = "dominant_unresolved_perceptual_degradation"
BAND_SPLIT_DIAGNOSTIC_STATUS = "partial_improvement_not_sufficient"
CROSSFADE_DIAGNOSTIC_STATUS = "not_dominant_cause"
GAUSSIAN_NOISE_DIAGNOSTIC_STATUS = "not_dominant_cause"
LOW_CEPSTRAL_ORDER_DIAGNOSTIC_STATUS = "not_dominant_cause"

# Calibrated owned excitation candidate. These paths contain no production weights and are not
# active until local owned-data calibration + held-out listening complete successfully.
GLOTTAL_CALIBRATION_MODULE = "lykenox_voice_engine/training/speech_glottal_calibration.py"
BAND_APERIODICITY_CALIBRATION_MODULE = (
    "lykenox_voice_engine/training/speech_band_aperiodicity_calibration.py"
)
CALIBRATED_EXCITATION_CANDIDATE_MODULE = (
    "lykenox_voice_engine/training/speech_vocoder_minimum_phase_glottal_excitation_v1.py"
)
CALIBRATED_EXCITATION_ORACLE_SCRIPT = "scripts/diagnostic_calibrated_glottal_oracle_v1.py"
GLOTTAL_CALIBRATION_ARTIFACT = "models/lykenox_identity/calibration/glottal_pulse_v1.json"
BAND_APERIODICITY_CALIBRATION_ARTIFACT = (
    "models/lykenox_identity/calibration/band_aperiodicity_v1.json"
)
CALIBRATED_EXCITATION_FAMILY = "owned_rosenberg_glottal_pulse_plus_measured_band_aperiodicity"
CALIBRATED_EXCITATION_IDENTITY_PARAMETERS_SOURCE = "owned_train_real_residual_measurements_only"
CALIBRATED_EXCITATION_STATUS = "implemented_awaiting_local_calibration_and_heldout_oracle_listening"
CALIBRATED_EXCITATION_PRODUCTION_ACTIVE = False
CALIBRATED_EXCITATION_GRADIENT_TRAINING_USED = False
CALIBRATED_EXCITATION_THIRD_PARTY_COMPONENT_USED = False
CALIBRATED_EXCITATION_FULL_HELDOUT_LISTENING_REQUIRED = True

# Bounded train/checkpoint creation stays blocked while the active production source is known to
# be perceptually defective and the calibrated replacement has not yet passed held-out listening.
TRAINING_BLOCKED_BY_SYNTHETIC_EXCITATION = True
BOUNDED_MODEL_OPTIMIZER_CURRENTLY_AUTHORIZED = False
SCOPED_NEW_CHECKPOINT_CURRENTLY_AUTHORIZED = False
PRODUCTION_RENDERER_MODIFICATION_AUTHORIZED_BY_THIS_EVIDENCE = False

NEXT_ACTION = "generate_owned_glottal_and_band_calibrations_then_run_calibrated_glottal_oracle_and_listen"
