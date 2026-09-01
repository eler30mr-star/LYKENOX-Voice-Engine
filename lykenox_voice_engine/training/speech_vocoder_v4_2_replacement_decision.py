"""Frozen engineering decision after direct v4.2/V8/V9 and owned-pipeline forensics.

LYKENOX is an identity-voice product intended for distribution. The vocoder architecture,
training state, and distributable model weights must remain LYKENOX-owned. Third-party
pretrained vocoder checkpoints are not an authorized product dependency, fallback, probe,
or replacement path.
"""

DECISION_VERSION = "vocoder-v4-2-replacement-decision-v15"
V4_2_ROLE = "intelligible_colored_baseline_only"
V4_2_FURTHER_TRAINING_AUTHORIZED = False
ACOUSTIC_TRAINING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_EQ_DENOISE_AUTHORIZED = False
SCRATCH_VOCODER_ITERATION_AUTHORIZED = False

VOCODER_OWNERSHIP_CONTRACT = "lykenox_owned_architecture_and_weights_only"
THIRD_PARTY_PRETRAINED_VOCODER_AUTHORIZED = False
THIRD_PARTY_VOCODER_CHECKPOINT_AUTHORIZED = False
DISTRIBUTION_REQUIRES_LYKENOX_OWNED_WEIGHTS = True

# The waveform-space v1 loss-weight contract remains frozen as valid historical evidence,
# but the minimum-phase Jacobian audit proved it is not authority-compatible for training
# this architecture. Only a read-only cepstrum-derived / parameter-cross-checked v2 weight
# recalibration is open. No optimizer, longer smoke, trainer, checkpoint, or persistent
# training is authorized.
LOSS_WEIGHT_CONTRACT_AUTHORIZED = True
LOSS_V2_WEIGHT_CONTRACT_FROZEN = True
LOSS_V2_WEIGHT_CONTRACT_V1_ARCHITECTURE_COMPATIBLE = False
ACTIVE_ARCHITECTURE_WEIGHT_CONTRACT_AUTHORIZED = False
MODEL_INSTANTIATION_AUTHORIZED = True
FRAME_RATE_CEPSTRAL_PREDICTOR_IMPLEMENTATION_AUTHORIZED = True
BOUNDED_OPTIMIZER_SMOKE_AUTHORIZED = False
BOUNDED_OPTIMIZER_SMOKE_MAX_UPDATES = 2
BOUNDED_OPTIMIZER_SMOKE_SEGMENT_FRAMES = 32
BOUNDED_OPTIMIZER_SMOKE_MAX_ITEMS = 1
BOUNDED_OPTIMIZER_SMOKE_STATUS = "pass"
BOUNDED_OPTIMIZER_SMOKE_CONSUMED = True
PARAMETER_SPACE_GRADIENT_AUDIT_AUTHORIZED = False
PARAMETER_SPACE_GRADIENT_AUDIT_STATUS = "pass"
ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_AUTHORIZED = True
ARCHITECTURE_WEIGHT_CONTRACT_V2_AUTHORIZED = False
EXTENDED_TRAINABILITY_SMOKE_AUTHORIZED = False
OPTIMIZER_CREATION_AUTHORIZED = False
TRAINER_IMPLEMENTATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_ARCHITECTURE_AUTHORIZED = False
NEW_VOCODER_CHECKPOINT_AUTHORIZED = False
VOCODER_ARCHITECTURE_SELECTION_AUTHORIZED = True

OWNED_VOCODER_DATA_CONTRACT = (
    "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
)
OWNED_VOCODER_LOSS_CONTRACT = (
    "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
)
OWNED_VOCODER_PRESENCE_CONTRACT = (
    "owned-vocoder-presence-v2-valid-context-target-relative"
)
OWNED_VOCODER_LOSS_WEIGHT_CONTRACT = "owned-vocoder-loss-v2-weight-contract-v1"
OWNED_VOCODER_ARCHITECTURE_CONTRACT = "owned-vocoder-architecture-contract-v1"
OWNED_STATIC_RENDERER = "owned-minimum-phase-time-varying-renderer-v1"
OWNED_FRAME_RATE_PREDICTOR = "lykenox_owned_frame_rate_cepstral_predictor_v1"
ARCHITECTURE_WEIGHT_RECALIBRATION_AUDIT_VERSION = (
    "owned-minimum-phase-architecture-weight-recalibration-audit-v1"
)
ARCHITECTURE_WEIGHT_RECALIBRATION_CANDIDATE_VERSION = (
    "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2-candidate"
)
HISTORICAL_PRESENCE_EDGE_SEMANTICS_REJECTED = True

FORENSIC_BASELINE = {
    "spectral_centroid_relative_pct": -16.468669,
    "spectral_envelope_300_4k_l1_db": 3.606936,
    "presence_1k_8k_error_db": 2.604178,
    "mean_abs_voiced_pitch_delta_cents": 35.771017,
}

ABLATION_VERDICT = (
    "No v4.2 source ablation improved the renderer. Removing harmonic, aperiodic, or all "
    "explicit excitation worsened envelope/presence/pitch error, while mel-envelope-only "
    "collapsed. v4.2 therefore depends on explicit excitation to speak but remains "
    "spectrally colored with all paths active."
)

V8_VERDICT = (
    "V8 fixed STFT/iSTFT analysis-synthesis is numerically sound, but predicting absolute "
    "complex STFT coefficients learned severe hop-locked repetition relative to the paired "
    "reference (+0.341784 hop and +0.671333 double-hop autocorrelation excess). V8 is "
    "architecturally rejected and persistent training is forbidden."
)

V9_VERDICT = (
    "V9 removed V8 frame-grid excess and its differential spectral factorization was "
    "numerically sound, but the bounded oracle output was not usable speech. V9 is "
    "perceptually rejected and persistent training is forbidden."
)

CONDITIONING_FORENSICS_STATUS = "pass"
CONDITIONING_FORENSIC_METRICS = {
    "mean_all_voicing_disagreement_fraction": 0.026042,
    "mean_all_periodicity_l1": 0.002821,
    "mean_all_f0_mae_cents_on_common_voiced": 5.12231,
    "mean_boundary_voicing_disagreement_fraction": 0.166667,
    "mean_boundary_periodicity_l1": 0.045135,
    "mean_boundary_f0_mae_cents_on_common_voiced": 79.25969,
    "mean_interior_voicing_disagreement_fraction": 0.016667,
    "mean_interior_periodicity_l1": 0.0,
    "mean_interior_f0_mae_cents_on_common_voiced": 0.0,
}
CONDITIONING_FORENSIC_FINDING = (
    "Historical v1 crop-local pitch conditioning was measurably inconsistent with the "
    "owned full-utterance pitch cache, predominantly at crop boundaries: 16.6667% mean "
    "boundary voicing disagreement and 79.25969 cents mean boundary F0 error, versus 0 "
    "cents and exact periodicity in the interior. The corrected v2 contract was verified "
    "as an exact slice of the same hashed full-utterance pitch cache used by the acoustic "
    "model. V1 remains historical only; all future vocoder work must use v2."
)

LOSS_EDGE_FORENSICS_STATUS = "pass"
LOSS_EDGE_FORENSIC_METRICS = {
    "mel_crop_local_frame_count": 65,
    "mel_conditioning_frame_count": 64,
    "mel_extra_terminal_frame_without_conditioning": True,
    "mean_mel_all_log_l1": 0.00786993,
    "mean_mel_artificial_log_l1": 0.16789168,
    "mean_mel_interior_log_l1": 0.00000001,
    "stft_256_64_mean_artificial_context_fraction": 0.0155642,
    "stft_256_64_mean_artificial_log_magnitude_l1": 0.5867609,
    "stft_256_64_mean_interior_log_magnitude_l1": 0.0,
    "stft_512_128_mean_artificial_context_fraction": 0.03100775,
    "stft_512_128_mean_artificial_log_magnitude_l1": 0.57506599,
    "stft_512_128_mean_interior_log_magnitude_l1": 0.0,
    "stft_1024_256_mean_artificial_context_fraction": 0.06153846,
    "stft_1024_256_mean_artificial_log_magnitude_l1": 0.5766321,
    "stft_1024_256_mean_interior_log_magnitude_l1": 0.0,
}
LOSS_EDGE_FORENSIC_FINDING = (
    "Historical centered crop-local spectral losses supervised artificial reflected context "
    "at crop boundaries even though full-utterance context exists in the owned dataset. "
    "Across all three STFT resolutions the measured interior log-magnitude discrepancy "
    "was exactly 0 while artificial boundary frames were about 0.57 log-L1. The historical "
    "crop-local mel analysis also produced 65 frames for 64 conditioning frames, including "
    "one terminal frame with no conditioning authority. Future owned vocoder objectives "
    "must score only centered frames with complete crop context and compare generated mel "
    "directly to the cached 64-frame conditioning grid. Historical V1 losses remain frozen "
    "for reproducibility only."
)

LOSS_V2_TARGET_CONSISTENCY_STATUS = "pass"
LOSS_V2_TARGET_CONSISTENCY_METRICS = {
    "exact_conditioning_frame_contract": True,
    "target_reconstruction_exact_on_valid_context": True,
    "conditioning_envelope_exact_on_valid_context": True,
    "mean_reconstruction_target_self_total": 0.0,
    "mean_reconstruction_target_self_log_magnitude": 0.0,
    "mean_conditioning_aligned_envelope_total": 0.0000000263,
    "mean_conditioning_aligned_log_mel_l1": 0.000000012,
    "mean_conditioning_aligned_spectral_slope_l1": 0.0000000188,
    "mean_conditioning_aligned_temporal_delta_l1": 0.0000000197,
    "conditioning_frames": 64,
    "analysis_frames": 65,
    "valid_conditioning_frames": 61,
    "reconstruction_valid_frame_counts": (253, 125, 61),
    "reconstruction_analysis_frame_counts": (257, 129, 65),
}
LOSS_V2_TARGET_CONSISTENCY_FINDING = (
    "Owned Loss V2 passed the real-data target-consistency contract. Target waveform "
    "against itself is exactly zero on valid-context multi-resolution reconstruction, and "
    "the target waveform against its owned cached conditioning mel is numerically zero "
    "(~1e-8) on the 61 valid conditioning frames. The 65th crop-local mel frame is excluded "
    "because no conditioning slot exists for it. This validates objective semantics only; "
    "it does not authorize persistent training."
)

LOSS_V2_GRADIENT_BALANCE_STATUS = "pass"
LOSS_V2_GRADIENT_BALANCE_METRICS = {
    "mean_reconstruction_gradient_norm": 11.3107378483,
    "mean_envelope_gradient_norm": 3.5935441388,
    "mean_spectral_balance_gradient_norm": 0.1855753782,
    "mean_reference_weighted_reconstruction_share": 0.854356,
    "mean_reference_weighted_envelope_share": 0.142148,
    "mean_reference_weighted_spectral_balance_share": 0.003496,
    "mean_reconstruction_vs_envelope_cosine": 0.344321,
    "mean_reconstruction_vs_spectral_balance_cosine": 0.039248,
    "mean_envelope_vs_spectral_balance_cosine": 0.075625,
    "minimum_pairwise_gradient_cosine": -0.001014,
    "mean_combined_reconstruction_alignment": 0.988408,
    "mean_combined_envelope_alignment": 0.4742,
    "mean_combined_spectral_balance_alignment": 0.050277,
    "minimum_combined_gradient_alignment_cosine": 0.005388,
    "maximum_reference_weighted_gradient_norm_share": 0.929409,
}
LOSS_V2_GRADIENT_BALANCE_FINDING = (
    "The three-objective diagnostic gradient audit passed numerically and showed no strong "
    "directional conflict, but the historical reference weights are rejected for future "
    "work. Reconstruction carried 85.4356% mean weighted gradient authority and reached "
    "92.9409%, while spectral balance carried only 0.3496% mean authority."
)

LOSS_V2_FOUR_OBJECTIVE_CALIBRATION_STATUS = "pass"
LOSS_V2_FOUR_OBJECTIVE_CALIBRATION_METRICS = {
    "mean_reconstruction_gradient_norm": 11.3107378483,
    "mean_envelope_gradient_norm": 3.5935441388,
    "mean_presence_gradient_norm": 0.5849306592,
    "mean_spectral_balance_gradient_norm": 0.1855753782,
    "derived_reconstruction_weight": 1.0,
    "derived_envelope_weight": 3.1475160486,
    "derived_presence_weight": 19.3368866395,
    "derived_spectral_balance_weight": 60.9495610684,
    "mean_reconstruction_share": 0.266542,
    "mean_envelope_share": 0.26621,
    "mean_presence_share": 0.231574,
    "mean_spectral_balance_share": 0.235674,
    "minimum_reconstruction_share": 0.16015,
    "minimum_envelope_share": 0.119819,
    "minimum_presence_share": 0.102618,
    "minimum_spectral_balance_share": 0.12295,
    "maximum_reconstruction_share": 0.526076,
    "maximum_envelope_share": 0.378526,
    "maximum_presence_share": 0.429119,
    "maximum_spectral_balance_share": 0.36469,
    "minimum_reconstruction_combined_alignment": 0.34519,
    "minimum_envelope_combined_alignment": 0.318401,
    "minimum_presence_combined_alignment": 0.357462,
    "minimum_spectral_balance_combined_alignment": 0.351216,
    "minimum_reconstruction_descent_dot": 58.8145828247,
    "minimum_envelope_descent_dot": 12.9584732056,
    "minimum_presence_descent_dot": 0.9880405664,
    "minimum_spectral_balance_descent_dot": 0.3690142632,
    "maximum_derived_weighted_gradient_norm_share": 0.526076,
}
LOSS_V2_FOUR_OBJECTIVE_CALIBRATION_FINDING = (
    "The valid-context four-objective calibration passed and removed the historical "
    "reconstruction monopoly. Mean weighted gradient authority is distributed across "
    "reconstruction/envelope/presence/spectral balance at roughly 23-27%, every objective "
    "retains meaningful authority, and all measured combined alignments and first-order "
    "descent dots are positive."
)

LOSS_V2_WEIGHT_CONTRACT_CANDIDATE = {
    "reconstruction": 1.0,
    "envelope": 3.1475,
    "presence": 19.3369,
    "spectral_balance": 60.9496,
}
LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_STATUS = "pass"
LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_REQUIRED = False
LOSS_V2_WEIGHT_CONTRACT_SENSITIVITY_METRICS = {
    "scenario_count": 23,
    "relative_weight_perturbation": 0.10,
    "candidate_derivation_relative_errors": {
        "reconstruction": 0.0,
        "envelope": 0.0000051,
        "presence": 0.00000069,
        "spectral_balance": 0.00000064,
    },
    "baseline_minimum_weighted_gradient_norm_shares": {
        "reconstruction": 0.16015,
        "envelope": 0.119818,
        "presence": 0.102618,
        "spectral_balance": 0.122951,
    },
    "all_scenarios_minimum_weighted_gradient_norm_shares": {
        "reconstruction": 0.134961,
        "envelope": 0.100216,
        "presence": 0.085556,
        "spectral_balance": 0.102896,
    },
    "baseline_minimum_combined_gradient_alignment_cosines": {
        "reconstruction": 0.34519,
        "envelope": 0.3184,
        "presence": 0.357462,
        "spectral_balance": 0.351217,
    },
    "all_scenarios_minimum_combined_gradient_alignment_cosines": {
        "reconstruction": 0.306252,
        "envelope": 0.271811,
        "presence": 0.310115,
        "spectral_balance": 0.302829,
    },
    "all_scenarios_minimum_first_order_descent_dots": {
        "reconstruction": 57.8851928711,
        "envelope": 10.8785772324,
        "presence": 0.8195143342,
        "spectral_balance": 0.3058912754,
    },
    "baseline_maximum_weighted_gradient_norm_share": 0.526076,
    "all_scenarios_maximum_weighted_gradient_norm_share": 0.575682,
    "candidate_tracks_derivation": True,
    "authority_retained": True,
    "alignment_positive": True,
    "alignment_retained": True,
    "descent_positive": True,
    "dominance_bounded": True,
}
LOSS_V2_WEIGHT_CONTRACT_FROZEN_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 3.1475,
    "presence": 19.3369,
    "spectral_balance": 60.9496,
}
LOSS_V2_WEIGHT_CONTRACT_FINDING = (
    "The data-derived four-objective weight vector survived 23 bounded sensitivity "
    "scenarios spanning +/-10% relative perturbations. Across all scenarios every objective "
    "retained at least 8.5556% weighted-gradient authority, every combined-gradient "
    "alignment remained positive (minimum 0.271811), every first-order descent dot remained "
    "positive, and maximum single-objective authority remained bounded at 57.5682%. The "
    "rounded candidate tracks the rederived equalization weights to far below 0.5% relative "
    "error. The weight contract is therefore frozen as v1 in its original waveform-space "
    "calibration. The later architecture Jacobian audit does not alter that historical fact; "
    "it proves v1 must not be used as the active training weights for the minimum-phase path."
)

ARCHITECTURE_CONTRACT_STATUS = "pass"
ARCHITECTURE_CONTRACT_VALIDATION_TEST_COUNT = 21
STATIC_RENDERER_SAFETY_STATUS = "pass"
STATIC_RENDERER_SAFETY_AUDIT_VERSION = "owned-minimum-phase-renderer-safety-audit-v1"
STATIC_RENDERER_SAFETY_TEST_COUNT = 24
STATIC_RENDERER_SAFETY_METRICS = {
    "maximum_log_magnitude_factorization_error": 1.0685896612017132e-15,
    "maximum_reference_oracle_roundtrip_error": 6.938893903907228e-18,
    "flat_envelope_max_abs_identity_error": 0.0,
    "attenuating_filter_measured_rms_ratio": 0.002478752176666358,
    "attenuating_filter_max_abs_expected_error": 8.673617379884035e-19,
    "unvoiced_hop_autocorrelation_excess": -0.001309794233417115,
    "unvoiced_double_hop_autocorrelation_excess": -0.0013493497067419598,
    "unvoiced_grid_harmonic_power_fraction_excess": 0.00012359877649464196,
    "voiced_hop_autocorrelation_excess": 9.394839821898096e-05,
    "voiced_double_hop_autocorrelation_excess": 0.0005020093189166902,
    "voiced_grid_harmonic_power_fraction_excess": 0.00040792484893605215,
    "same_seed_max_abs_error": 0.0,
    "source_bypass_absent": True,
    "exact_output_length": True,
}
STATIC_RENDERER_SAFETY_FINDING = (
    "The owned fixed minimum-phase renderer passed its structural safety gate before any "
    "neural model existed. Cepstral factorization and the reference-envelope roundtrip are "
    "numerically exact, a flat envelope is exact identity, an attenuating filter suppresses "
    "the excitation without a source bypass, output length is exactly frames*256, same-seed "
    "excitation is deterministic, and paired voiced/unvoiced grid-excess metrics remain far "
    "below the severe-grid thresholds."
)

FRAME_RATE_PREDICTOR_STRUCTURAL_STATUS = "pass"
FRAME_RATE_PREDICTOR_STRUCTURAL_SMOKE_VERSION = "owned-frame-rate-cepstral-predictor-smoke-v1"
FRAME_RATE_PREDICTOR_STRUCTURAL_TEST_COUNT = 36
FRAME_RATE_PREDICTOR_STRUCTURAL_METRICS = {
    "predictor_output_shape": (2, 48, 64),
    "maximum_abs_initial_cepstrum": 0.0,
    "renderer_identity_max_abs_error": 0.0,
    "expected_waveform_samples": 12288,
    "actual_waveform_samples": 12288,
    "hop_autocorrelation_excess": 0.0,
    "double_hop_autocorrelation_excess": 0.0,
    "grid_harmonic_power_fraction_excess": 0.0,
    "parameter_count": 236736,
    "connected_nonzero_gradient_tensor_count": 30,
    "trainable_parameter_tensor_count": 30,
}
FRAME_RATE_PREDICTOR_STRUCTURAL_FINDING = (
    "The owned frame-rate cepstral predictor passed its pre-optimizer structural smoke. "
    "Its zero-initialized output is exactly neutral, the fixed renderer remains exact "
    "identity with exact frames*256 length, no frame-grid excess is introduced at neutral "
    "initialization, and all 30 trainable parameter tensors receive finite non-zero gradient "
    "in the deterministic connectivity probe."
)

BOUNDED_OPTIMIZER_SMOKE_VERSION = "owned-minimum-phase-bounded-optimizer-smoke-v1"
BOUNDED_OPTIMIZER_SMOKE_TEST_COUNT = 33
BOUNDED_OPTIMIZER_SMOKE_METRICS = {
    "utterance_id": "speech_0021_6cd35984e877_seg_002",
    "start_frame": 717,
    "segment_mel_frames": 32,
    "update_count": 2,
    "learning_rate": 0.0002,
    "max_gradient_norm": 1.0,
    "initial_total": 195.1538543701,
    "final_total": 194.9214477539,
    "relative_total_change": -0.0011908892,
    "initial_reconstruction": 19.2587738037,
    "final_reconstruction": 19.2581729889,
    "initial_envelope": 4.7642612457,
    "final_envelope": 4.7644839287,
    "initial_presence": 3.1395702362,
    "final_presence": 3.1374154091,
    "initial_spectral_balance": 1.6438173056,
    "final_spectral_balance": 1.640686512,
    "update_1_raw_gradient_norm": 580.735168457,
    "update_2_raw_gradient_norm": 580.9564819336,
    "parameter_delta_norm": 0.0004,
    "parameter_delta_max_abs": 0.0000964731,
    "final_hop_autocorrelation_excess": -0.00002784491516649723,
    "final_double_hop_autocorrelation_excess": 0.00002341344952583313,
    "final_grid_harmonic_power_fraction_excess": 0.000048296526074409485,
    "severe_grid_excess": False,
    "checkpoints_unchanged": True,
}
BOUNDED_OPTIMIZER_SMOKE_FINDING = (
    "The exactly-two-update owned real-data smoke passed: the frozen Loss V2 total decreased "
    "by 0.1191%, parameters moved by L2=0.0004, exact output length and grid safety remained "
    "intact, and protected checkpoints were unchanged. However, raw parameter-gradient norms "
    "were about 581 against a clip limit of 1.0 on both steps, and the envelope term increased "
    "slightly while the total decreased. This required the later Jacobian authority audit."
)

PARAMETER_SPACE_GRADIENT_AUDIT_VERSION = (
    "owned-minimum-phase-parameter-gradient-authority-audit-v1"
)
PARAMETER_SPACE_GRADIENT_AUDIT_TEST_COUNT = 41
PARAMETER_SPACE_GRADIENT_AUDIT_PROBE_COUNT = 8
PARAMETER_SPACE_GRADIENT_AUDIT_METRICS = {
    "neutral_mean_weighted_gradient_norm_shares": {
        "reconstruction": 0.046928433266268124,
        "envelope": 0.009529865430589276,
        "presence": 0.234379802425534,
        "spectral_balance": 0.7091618988776086,
    },
    "neutral_minimum_combined_gradient_alignment_cosines": {
        "reconstruction": 0.08097482472658157,
        "envelope": -0.3273468315601349,
        "presence": 0.8204574584960938,
        "spectral_balance": 0.976812481880188,
    },
    "neutral_minimum_first_order_descent_dots": {
        "reconstruction": 478.39349365234375,
        "envelope": -387.89593505859375,
        "presence": 2462.26708984375,
        "spectral_balance": 2270.551025390625,
    },
    "neutral_mean_combined_gradient_norm": 524.2026596069336,
    "neutral_mean_clip_scale_if_max_norm_1": 0.0019496183077621803,
    "neutral_maximum_weighted_gradient_norm_share": 0.765470299656555,
    "connected_mean_weighted_gradient_norm_shares": {
        "reconstruction": 0.046930590150991255,
        "envelope": 0.009529655236837993,
        "presence": 0.2343777644546192,
        "spectral_balance": 0.7091619901575515,
    },
    "connected_minimum_envelope_alignment": -0.32732290029525757,
    "connected_minimum_envelope_descent_dot": -387.87164306640625,
    "connected_mean_combined_gradient_norm": 524.2039108276367,
    "connected_mean_clip_scale_if_max_norm_1": 0.00194961708097689,
    "connected_maximum_weighted_gradient_norm_share": 0.7654694679457218,
    "cepstrum_neutral_mean_weighted_gradient_norm_shares": {
        "reconstruction": 0.04117481310162209,
        "envelope": 0.008032201568982572,
        "presence": 0.1949435194527072,
        "spectral_balance": 0.7558494658766881,
    },
    "cepstrum_neutral_minimum_envelope_alignment": -0.2579599916934967,
    "cepstrum_connected_minimum_envelope_alignment": -0.2580244839191437,
}
PARAMETER_SPACE_GRADIENT_AUDIT_FINDING = (
    "The read-only Jacobian audit passed as a valid measurement but rejected waveform-space "
    "weight contract v1 for active minimum-phase training. In cepstrum space spectral balance "
    "holds 75.5849% mean weighted authority while envelope holds 0.8032%, and envelope can be "
    "anti-aligned with the combined direction (minimum cosine about -0.258). Parameter space "
    "repeats the failure: spectral balance about 70.9162%, envelope about 0.9530%, minimum "
    "envelope alignment about -0.327 and negative first-order descent dot. Because the defect "
    "already exists before the predictor, LR/clip changes cannot repair the objective geometry."
)

NEXT_ARCHITECTURE = "owned_minimum_phase_time_varying_filter_over_neutral_excitation"
NEXT_GATE = (
    "audit_owned_minimum_phase_architecture_coupled_loss_v2_weight_recalibration"
)
