"""Frozen engineering decision after direct v4.2/V8/V9 vocoder forensics.

LYKENOX is an identity-voice product intended for distribution. The vocoder architecture,
training state, and distributable model weights must remain LYKENOX-owned. Third-party
pretrained vocoder checkpoints are not an authorized product dependency, fallback, probe,
or replacement path.
"""

DECISION_VERSION = "vocoder-v4-2-replacement-decision-v6"
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
OWNED_VOCODER_DATA_CONTRACT = (
    "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
)

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

NEXT_ARCHITECTURE = "undecided_after_owned_pipeline_forensics"
NEXT_GATE = "audit_owned_vocoder_loss_edge_and_objective_semantics"
