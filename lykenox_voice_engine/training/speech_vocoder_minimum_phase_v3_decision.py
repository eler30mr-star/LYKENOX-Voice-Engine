"""Current minimum-phase vocoder decision after residual and excitation diagnostics.

Historical evidence is preserved without rewriting rejected candidates. The v2 equal-norm loss
weights remain rejected for directional conflict. The minimum-phase envelope/filter path remains
supported by the positive real-residual oracle: when the owned real residual replaces synthetic
excitation, complete held-out resynthesis was reported clean and natural, matching the original
voice audio.

The calibrated Rosenberg pulse + measured four-band aperiodicity candidate is now also rejected.
The owner reports that a local calibration based on 97,168 pitch-synchronous cycles still produced
gangoso/rough held-out oracle audio. This moves the active engineering problem from pulse tuning to
predicting the owned residual/excitation detail itself while keeping the proven filter path fixed.
"""

DECISION_VERSION = "owned-minimum-phase-v3-decision-v4-residual-predictor-pivot"
POLICY_ID = "LYX-POL-001"
SUPERSEDES_GATE_STATE_FROM = "owned-vocoder-architecture-contract-v1"
ARCHITECTURE_FAMILY = "owned_minimum_phase_filter_over_owned_predicted_residual_candidate"

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

# Positive pure-DSP isolation evidence.
REAL_RESIDUAL_DIAGNOSTIC_SCRIPT = "scripts/diagnostic_real_residual_resynthesis_v1.py"
REAL_RESIDUAL_EVIDENCE_DOC = "docs/LYKENOX_VOCODER_MINIMUM_PHASE_REAL_RESIDUAL_EVIDENCE.md"
REAL_RESIDUAL_MODEL_USED = False
REAL_RESIDUAL_TRAINING_EXECUTED = False
REAL_RESIDUAL_CHECKPOINT_USED = False
REAL_RESIDUAL_SYNTHETIC_EXCITATION_USED = False
REAL_RESIDUAL_HUMAN_LISTENING_RESULT = "clean_natural_matches_original_voice_audio"
REAL_RESIDUAL_FILTER_ENVELOPE_PATH_STATUS = "clean_oracle_resynthesis_demonstrated"

# Rejected synthetic-excitation evidence.
BAND_SPLIT_DIAGNOSTIC_STATUS = "partial_improvement_not_sufficient"
CROSSFADE_DIAGNOSTIC_STATUS = "not_dominant_cause"
GAUSSIAN_NOISE_DIAGNOSTIC_STATUS = "not_dominant_cause"
LOW_CEPSTRAL_ORDER_DIAGNOSTIC_STATUS = "not_dominant_cause"
CALIBRATED_GLOTTAL_REJECTION_DOC = "docs/LYKENOX_VOCODER_CALIBRATED_GLOTTAL_REJECTION.md"
CALIBRATED_EXCITATION_FAMILY = "owned_rosenberg_glottal_pulse_plus_measured_band_aperiodicity"
CALIBRATED_EXCITATION_REPORTED_CYCLE_COUNT = 97168
CALIBRATED_EXCITATION_LOCAL_LISTENING_RESULT = "gangoso_rough_rejected"
CALIBRATED_EXCITATION_STATUS = "rejected_perceptual_structural_limit"
CALIBRATED_EXCITATION_PRODUCTION_ACTIVE = False
SYNTHETIC_PARAMETRIC_EXCITATION_STATUS = "rejected_as_dominant_quality_path"

# CELP/codebook clarification for TTS.
PURE_CELP_ANALYSIS_BY_SYNTHESIS_SELECTED = False
PURE_CELP_REASON = (
    "inference_has_no_target_residual_for_codebook_search; any codebook path still needs an owned selector"
)
OWNED_RESIDUAL_CODEBOOK_ALLOWED_AS_INTERNAL_REPRESENTATION = True
OWNED_RESIDUAL_SELECTOR_MUST_BE_LYKENOX_TRAINED = True

# Active next candidate: narrowly scoped excitation/residual prediction only.
RESIDUAL_PREDICTOR_CANDIDATE_SELECTED = True
RESIDUAL_PREDICTOR_SCOPE = "predict_owned_real_residual_detail_only"
RESIDUAL_TARGET_SOURCE = "owned_real_residual_extracted_with_step3f_method"
MINIMUM_PHASE_FILTER_PATH_FROZEN = True
RESIDUAL_PREDICTOR_CPU_ONLY = True
RESIDUAL_PREDICTOR_THIRD_PARTY_WEIGHTS_ALLOWED = False
RESIDUAL_PREDICTOR_PRODUCT_ACCEPTANCE_REQUIRES_FULL_HELDOUT_LISTENING = True

# Existing bounded end-to-end training/checkpoint creation stays blocked until the residual predictor
# has its own target contract, bounded CPU smoke and oracle/full-utterance evidence.
TRAINING_BLOCKED_BY_SYNTHETIC_EXCITATION = True
BOUNDED_MODEL_OPTIMIZER_CURRENTLY_AUTHORIZED = False
SCOPED_NEW_CHECKPOINT_CURRENTLY_AUTHORIZED = False
PRODUCTION_RENDERER_MODIFICATION_AUTHORIZED_BY_THIS_EVIDENCE = False

NEXT_ACTION = "define_owned_real_residual_target_and_small_cpu_residual_predictor_then_oracle_validate_before_end_to_end_training"
