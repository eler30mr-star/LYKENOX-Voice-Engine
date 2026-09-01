"""Frozen engineering decision after direct v4.2/V8/V9 and owned-pipeline forensics.

LYKENOX is an identity-voice product intended for distribution. The vocoder architecture,
training state, and distributable model weights must remain LYKENOX-owned. Third-party
pretrained vocoder checkpoints are not an authorized product dependency, fallback, probe,
or replacement path.
"""

DECISION_VERSION = "vocoder-v4-2-replacement-decision-v9"
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
NEW_VOCODER_ARCHITECTURE_AUTHORIZED = False
LOSS_WEIGHT_CONTRACT_AUTHORIZED = False
OWNED_VOCODER_DATA_CONTRACT = (
    "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
)
OWNED_VOCODER_LOSS_CONTRACT = (
    "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
)
OWNED_VOCODER_PRESENCE_CONTRACT = (
    "owned-vocoder-presence-v2-valid-context-target-relative"
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
    "it does not authorize loss weights, a new architecture, or persistent training."
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
    "directional conflict, but the historical reference weights are not an acceptable "
    "future contract. Reconstruction carried 85.4356% mean weighted gradient authority "
    "and reached 92.9409%, while spectral balance carried only 0.3496% mean authority. "
    "The historical presence objective cannot simply be added because its centered crop "
    "STFT includes the same artificial edge context already rejected by Loss V2. A new "
    "valid-context Presence V2 must therefore participate in a four-objective, data-derived "
    "gradient calibration before any loss weights or architecture can be authorized."
)

NEXT_ARCHITECTURE = "undecided_after_owned_pipeline_forensics"
NEXT_GATE = "audit_owned_vocoder_four_objective_gradient_calibration_before_weight_contract"
