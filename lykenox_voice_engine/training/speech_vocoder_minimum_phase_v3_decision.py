"""Current minimum-phase vocoder decision after residual and excitation diagnostics.

Historical evidence is preserved without rewriting rejected candidates. The v2 equal-norm loss
weights remain rejected for directional conflict. The minimum-phase envelope/filter path remains
supported by the positive real-residual oracle: when the owned real residual replaces synthetic
excitation, complete held-out resynthesis was reported clean and natural, matching the original
voice audio.

The calibrated Rosenberg pulse + measured four-band aperiodicity candidate is also rejected. The
owner reports that a local calibration based on 97,168 pitch-synchronous cycles still produced
gangoso/rough held-out oracle audio.

The CELP-style codebook line now has a decisive representation isolation result. V1 was too quiet to
judge because of an arbitrary oracle-gain cap. V2 corrected level but independently substituted each
held-out residual window with a train codeword and remained intelligible but gangoso. The identity
roundtrip then analyzed the exact clean held-out real residual into the same 512-sample / 256-hop
sqrt-Hann representation, resynthesized those exact vectors without substitution or gain change, and
the owner reports that the final filtered resynthesis again sounds correct and matches the original
voice audio. Therefore the codevector window/OLA representation and frozen minimum-phase filter are
exculpated when the correct residual trajectory is used.

V3 removed per-window polarity inversion and added complete-utterance residual-domain overlap
continuity, but the owner reports the final held-out waveform remains gangoso. Therefore residual-
domain codeword similarity/continuity is rejected as the selection criterion. V4 moved final
selection and non-negative gain into the exact local frozen-renderer waveform domain but keeps a
purely local greedy argmin per analysis window. V5 preserves V4 unchanged and adds deterministic
bounded beam search with continuity measured in that same filtered waveform domain over the exact
256-sample adjacent OLA overlap.

No selector training is authorized. No further algorithm iteration is authorized until V5 is run and
the owner reports complete held-out listening against V4 and the clean identity-roundtrip ceiling.
"""

DECISION_VERSION = "owned-minimum-phase-v3-decision-v11-codebook-synthesis-domain-coherent-oracle"
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

# CELP-style owned residual codebook capacity diagnostics.
RESIDUAL_CODEBOOK_DECISION_DOC = "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_ORACLE_DECISION.md"
RESIDUAL_CODEBOOK_EXECUTION_EVIDENCE_DOC = (
    "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_ORACLE_EXECUTION_EVIDENCE.md"
)
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_EVIDENCE_DOC = (
    "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_EVIDENCE.md"
)
RESIDUAL_CODEBOOK_V3_REJECTION_DOC = "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_V3_REJECTION.md"
RESIDUAL_CODEBOOK_V5_DECISION_DOC = (
    "docs/LYKENOX_VOCODER_RESIDUAL_CODEBOOK_V5_SYNTHESIS_DOMAIN_COHERENT.md"
)
RESIDUAL_CODEBOOK_MODULE = "lykenox_voice_engine/training/speech_residual_codebook_v1.py"
RESIDUAL_CODEBOOK_ORACLE_V1_SCRIPT = "scripts/diagnostic_residual_codebook_oracle_v1.py"
RESIDUAL_CODEBOOK_ORACLE_V2_SCRIPT = "scripts/diagnostic_residual_codebook_oracle_v2.py"
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_SCRIPT = (
    "scripts/diagnostic_residual_codebook_identity_roundtrip_v1.py"
)
RESIDUAL_CODEBOOK_ORACLE_V3_SEQUENCE_SCRIPT = (
    "scripts/diagnostic_residual_codebook_oracle_v3_sequence_coherent.py"
)
RESIDUAL_CODEBOOK_ORACLE_V4_SYNTHESIS_DOMAIN_SCRIPT = (
    "scripts/diagnostic_residual_codebook_oracle_v4_synthesis_domain.py"
)
RESIDUAL_CODEBOOK_ORACLE_V5_SYNTHESIS_DOMAIN_COHERENT_SCRIPT = (
    "scripts/diagnostic_residual_codebook_oracle_v5_synthesis_domain_coherent.py"
)
RESIDUAL_CODEBOOK_SOURCE = "owned_train_real_residual_only"
RESIDUAL_CODEBOOK_THIRD_PARTY_DATA_ALLOWED = False
RESIDUAL_CODEBOOK_THIRD_PARTY_MODEL_OR_CHECKPOINT_ALLOWED = False
RESIDUAL_CODEBOOK_REMOTE_INFERENCE_ALLOWED = False
RESIDUAL_CODEBOOK_PRODUCTION_ACTIVE = False
CELP_STYLE_ANALYSIS_BY_SYNTHESIS_ORACLE_ALLOWED = True
HELDOUT_RESIDUAL_ALLOWED_AS_ORACLE_SEARCH_TARGET_ONLY = True
HELDOUT_RESIDUAL_ALLOWED_IN_CODEBOOK = False
ORACLE_SELECTED_INDICES_OR_GAINS_VALID_FOR_PRODUCT_INFERENCE = False

# Owner-reported local codebook construction evidence from 2026-09-01.
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

# V2 corrected level but independent residual-domain substitution did not reach the clean ceiling.
RESIDUAL_CODEBOOK_ORACLE_V2_STATUS = "rejected_independent_window_substitution"
RESIDUAL_CODEBOOK_ORACLE_V2_SELECTION = "max_abs_normalized_correlation"
RESIDUAL_CODEBOOK_ORACLE_V2_GAIN = "signed_target_energy_over_codeword_energy"
RESIDUAL_CODEBOOK_ORACLE_V2_POSTHOC_OUTPUT_GAIN_NORMALIZATION = False
RESIDUAL_CODEBOOK_ORACLE_V2_ORACLE_GAIN_VALID_FOR_PRODUCT = False
RESIDUAL_CODEBOOK_ORACLE_V2_FINAL_LISTENING_RESULT = "intelligible_but_gangoso"
RESIDUAL_CODEBOOK_ORACLE_V2_SELECTED_RESIDUAL_LISTENING_RESULT = "high_pitched_excitation_no_speech"
RESIDUAL_CODEBOOK_ORACLE_V2_CODEBOOK_REJECTED = False

# Identity roundtrip result: the exact residual trajectory survives the codebook representation and
# returns clean speech after the frozen filter. The residual itself need not sound like speech.
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_STATUS = "pass_clean_final_resynthesis"
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_CODEWORD_SUBSTITUTION = False
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_ORACLE_GAIN = False
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_RESIDUAL_LISTENING_RESULT = "noisy_non_speech_like_expected_excitation"
RESIDUAL_CODEBOOK_IDENTITY_ROUNDTRIP_FINAL_LISTENING_RESULT = "correct_matches_original_voice_audio"
RESIDUAL_CODEBOOK_512_256_SQRT_HANN_REPRESENTATION_STATUS = "exculpated"
RESIDUAL_CODEBOOK_FROZEN_FILTER_STATUS = "exculpated_when_correct_residual_sequence_used"

# V3 used positive polarity plus residual-domain Viterbi overlap continuity and still sounded gangoso.
RESIDUAL_CODEBOOK_ORACLE_V3_SEQUENCE_STATUS = "rejected_perceptually_gangoso"
RESIDUAL_CODEBOOK_ORACLE_V3_PER_WINDOW_POLARITY_INVERSION_ALLOWED = False
RESIDUAL_CODEBOOK_ORACLE_V3_SEQUENCE_SEARCH = "viterbi_topk_positive_cosine_plus_overlap_continuity"
RESIDUAL_CODEBOOK_ORACLE_V3_FINAL_LISTENING_RESULT = "gangoso"
RESIDUAL_CODEBOOK_ORACLE_V3_SELECTED_RESIDUAL_LISTENING_RESULT = "noise_non_speech_like"
RESIDUAL_CODEBOOK_ORACLE_V3_TRAINING_USED = False
RESIDUAL_CODEBOOK_ORACLE_V3_PRODUCT_ACTIVE = False
RESIDUAL_CODEBOOK_ORACLE_V3_CODEBOOK_REJECTED = False
RESIDUAL_CODEBOOK_FAILURE_LOCALIZATION = "residual_domain_codeword_selection_is_perceptually_insufficient"

# V4 is preserved unchanged as the synthesis-domain greedy baseline for V5 comparison.
RESIDUAL_CODEBOOK_ORACLE_V4_SYNTHESIS_DOMAIN_STATUS = "implemented_baseline_preserved_for_v5_comparison"
RESIDUAL_CODEBOOK_ORACLE_V4_FINAL_SELECTION_DOMAIN = "exact_local_frozen_renderer_waveform_contribution"
RESIDUAL_CODEBOOK_ORACLE_V4_GAIN_DOMAIN = "exact_local_frozen_renderer_waveform_contribution"
RESIDUAL_CODEBOOK_ORACLE_V4_GAIN_NON_NEGATIVE = True
RESIDUAL_CODEBOOK_ORACLE_V4_SELECTION_MEMORY = "none_greedy_per_window_argmin"
RESIDUAL_CODEBOOK_ORACLE_V4_TRAINING_USED = False
RESIDUAL_CODEBOOK_ORACLE_V4_PRODUCT_ACTIVE = False
RESIDUAL_CODEBOOK_ORACLE_V4_ORACLE_PARAMETERS_VALID_FOR_PRODUCT = False

# V5 keeps V4's preselection, exact filtered response, and gain unchanged and adds bounded sequence
# search whose continuity term is measured in the same filtered waveform domain.
RESIDUAL_CODEBOOK_ORACLE_V5_STATUS = "implemented_awaiting_owner_complete_heldout_listening"
RESIDUAL_CODEBOOK_ORACLE_V5_PRESELECT = "v4_broad_signed_residual_cosine_unchanged"
RESIDUAL_CODEBOOK_ORACLE_V5_LOCAL_RESPONSE = "v4_exact_local_frozen_renderer_response_unchanged"
RESIDUAL_CODEBOOK_ORACLE_V5_GAIN = "v4_non_negative_filtered_domain_least_squares_unchanged"
RESIDUAL_CODEBOOK_ORACLE_V5_SEQUENCE_SEARCH = "deterministic_bounded_beam"
RESIDUAL_CODEBOOK_ORACLE_V5_DEFAULT_BEAM_SIZE = 8
RESIDUAL_CODEBOOK_ORACLE_V5_DEFAULT_CONTINUITY_WEIGHT = 1.0
RESIDUAL_CODEBOOK_ORACLE_V5_CONTINUITY_DOMAIN = "filtered_waveform_256_sample_adjacent_ola_overlap"
RESIDUAL_CODEBOOK_ORACLE_V5_TRAINING_USED = False
RESIDUAL_CODEBOOK_ORACLE_V5_PRODUCT_ACTIVE = False
RESIDUAL_CODEBOOK_ORACLE_V5_ORACLE_PARAMETERS_VALID_FOR_PRODUCT = False
FURTHER_CODEBOOK_ALGORITHM_ITERATION_BEFORE_V5_LISTENING_AUTHORIZED = False

PURE_CELP_PRODUCT_INFERENCE_SELECTED = False
OWNED_RESIDUAL_SELECTOR_MUST_BE_LYKENOX_TRAINED = True
RESIDUAL_SELECTOR_TRAINING_CURRENTLY_AUTHORIZED = False
RESIDUAL_PREDICTOR_CANDIDATE_SELECTED = False

TRAINING_BLOCKED_BY_SYNTHETIC_EXCITATION = True
BOUNDED_MODEL_OPTIMIZER_CURRENTLY_AUTHORIZED = False
SCOPED_NEW_CHECKPOINT_CURRENTLY_AUTHORIZED = False
PRODUCTION_RENDERER_MODIFICATION_AUTHORIZED_BY_THIS_EVIDENCE = False

NEXT_ACTION = "run_v5_then_listen_vs_v4_and_identity_roundtrip_ceiling_and_report_before_any_further_iteration"
