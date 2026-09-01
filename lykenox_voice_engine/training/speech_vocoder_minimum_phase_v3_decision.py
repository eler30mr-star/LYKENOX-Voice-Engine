"""Current minimum-phase vocoder decision after residual and excitation diagnostics.

Historical evidence is preserved without rewriting rejected candidates. The v2 equal-norm loss
weights remain rejected for directional conflict. The minimum-phase envelope/filter path remains
supported by the positive real-residual oracle: when the owned real residual replaces synthetic
excitation, complete held-out resynthesis was reported clean and natural, matching the original
voice audio.

The calibrated Rosenberg pulse + measured four-band aperiodicity candidate is also rejected. The
owner reports that a local calibration based on 97,168 pitch-synchronous cycles still produced
gangoso/rough held-out oracle audio.

The first CELP-style codebook capacity run completed with 6,234 owned-train codewords across 58
buckets, but its v1 listening output is not a valid perceptual gate because the diagnostic imposed
an arbitrary maximum oracle gain of 4.0 and the resulting audio was reported too quiet to judge.
The owned codebook itself is retained. A v2 level-valid oracle now separates codevector shape from
excitation gain by selecting the best normalized-correlation codevector and matching target residual
energy before the unchanged minimum-phase filter. No selector training is authorized until the owner
listens to all three v2 utterances against both the references and the clean real-residual ceiling.
"""

DECISION_VERSION = "owned-minimum-phase-v3-decision-v7-codebook-v2-level-valid-listening"
POLICY_ID = "LYX-POL-001"
SUPERSEDES_GATE_STATE_FROM = "owned-vocoder-architecture-contract-v1"
ARCHITECTURE_FAMILY = "owned_minimum_phase_filter_over_owned_residual_codebook_candidate"

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
MINIMUM_PHASE_FILTER_PATH_FROZEN = True

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

# CELP-style owned residual codebook capacity diagnostic.
RESIDUAL_CODEBOOK_DECISION_DOC = "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_ORACLE_DECISION.md"
RESIDUAL_CODEBOOK_EXECUTION_EVIDENCE_DOC = (
    "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_ORACLE_EXECUTION_EVIDENCE.md"
)
RESIDUAL_CODEBOOK_MODULE = "lykenox_voice_engine/training/speech_residual_codebook_v1.py"
RESIDUAL_CODEBOOK_ORACLE_V1_SCRIPT = "scripts/diagnostic_residual_codebook_oracle_v1.py"
RESIDUAL_CODEBOOK_ORACLE_V2_SCRIPT = "scripts/diagnostic_residual_codebook_oracle_v2.py"
RESIDUAL_CODEBOOK_SOURCE = "owned_train_real_residual_only"
RESIDUAL_CODEBOOK_THIRD_PARTY_DATA_ALLOWED = False
RESIDUAL_CODEBOOK_THIRD_PARTY_MODEL_OR_CHECKPOINT_ALLOWED = False
RESIDUAL_CODEBOOK_REMOTE_INFERENCE_ALLOWED = False
RESIDUAL_CODEBOOK_PRODUCTION_ACTIVE = False
RESIDUAL_CODEBOOK_ORACLE_CAPACITY_TEST_SELECTED = True
CELP_STYLE_ANALYSIS_BY_SYNTHESIS_ORACLE_ALLOWED = True
HELDOUT_RESIDUAL_ALLOWED_AS_ORACLE_SEARCH_TARGET_ONLY = True
HELDOUT_RESIDUAL_ALLOWED_IN_CODEBOOK = False
ORACLE_SELECTED_INDICES_OR_GAINS_VALID_FOR_PRODUCT_INFERENCE = False

# Owner-reported local execution evidence from 2026-09-01.
RESIDUAL_CODEBOOK_BUILD_STATUS = "built_from_owned_train_real_residual"
RESIDUAL_CODEBOOK_RETAINED_CODEWORD_COUNT = 6234
RESIDUAL_CODEBOOK_BUCKET_COUNT = 58
RESIDUAL_CODEBOOK_HELDOUT_ITEM_COUNT = 3
RESIDUAL_CODEBOOK_LOCAL_DEVICE = "cpu"
RESIDUAL_CODEBOOK_TRAINING_EXECUTED = False
RESIDUAL_CODEBOOK_OPTIMIZER_CREATED = False
RESIDUAL_CODEBOOK_CHECKPOINT_WRITTEN = False
RESIDUAL_CODEBOOK_SELECTOR_TRAINING_AUTHORIZED_BY_RUN = False

# V1 generated valid files but is invalid as a human listening gate because an arbitrary gain cap
# made the result too quiet to judge. The codebook artifact is not rejected by this failure.
RESIDUAL_CODEBOOK_ORACLE_V1_GAIN_CAP = 4.0
RESIDUAL_CODEBOOK_ORACLE_V1_LISTENING_RESULT = "too_quiet_to_judge"
RESIDUAL_CODEBOOK_ORACLE_V1_PERCEPTUAL_GATE_VALID = False
RESIDUAL_CODEBOOK_ORACLE_V1_CODEBOOK_REJECTED = False

# V2 removes the arbitrary cap and treats gain as an oracle excitation parameter before filtering.
# This is not post-hoc output normalization and cannot be reused in product inference.
RESIDUAL_CODEBOOK_ORACLE_V2_STATUS = "implemented_awaiting_owner_complete_heldout_listening"
RESIDUAL_CODEBOOK_ORACLE_V2_SELECTION = "max_abs_normalized_correlation"
RESIDUAL_CODEBOOK_ORACLE_V2_GAIN = "signed_target_energy_over_codeword_energy"
RESIDUAL_CODEBOOK_ORACLE_V2_POSTHOC_OUTPUT_GAIN_NORMALIZATION = False
RESIDUAL_CODEBOOK_ORACLE_V2_ORACLE_GAIN_VALID_FOR_PRODUCT = False
RESIDUAL_CODEBOOK_LISTENING_STATUS = "awaiting_owner_level_valid_v2_complete_heldout_listening"

# A literal CELP codec path is not selected for product inference because new-text TTS has no target
# residual available for analysis-by-synthesis search. If the codebook oracle succeeds, a separate
# LYKENOX-owned selector/gain predictor may be considered under a new explicit training gate.
PURE_CELP_PRODUCT_INFERENCE_SELECTED = False
OWNED_RESIDUAL_SELECTOR_MUST_BE_LYKENOX_TRAINED = True
RESIDUAL_SELECTOR_TRAINING_CURRENTLY_AUTHORIZED = False
RESIDUAL_PREDICTOR_CANDIDATE_SELECTED = False

# Existing end-to-end training/checkpoint creation remains blocked until the codebook itself proves
# sufficient audible held-out capacity. Metrics and successful execution can reject but cannot grant
# this gate; complete held-out listening is required under LYX-POL-001.
TRAINING_BLOCKED_BY_SYNTHETIC_EXCITATION = True
BOUNDED_MODEL_OPTIMIZER_CURRENTLY_AUTHORIZED = False
SCOPED_NEW_CHECKPOINT_CURRENTLY_AUTHORIZED = False
PRODUCTION_RENDERER_MODIFICATION_AUTHORIZED_BY_THIS_EVIDENCE = False

NEXT_ACTION = "run_level_valid_residual_codebook_oracle_v2_and_listen_before_training_any_selector"
