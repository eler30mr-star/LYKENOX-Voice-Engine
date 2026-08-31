"""Frozen engineering decision after direct v4.2/V8/V9 vocoder forensics."""

DECISION_VERSION = "vocoder-v4-2-replacement-decision-v3"
V4_2_ROLE = "intelligible_colored_baseline_only"
V4_2_FURTHER_TRAINING_AUTHORIZED = False
ACOUSTIC_TRAINING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_EQ_DENOISE_AUTHORIZED = False
SCRATCH_VOCODER_ITERATION_AUTHORIZED = False

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
    "numerically sound, but the bounded oracle output was not usable speech. The bounded "
    "smoke itself was only a short trainability probe and must not be used to justify more "
    "scratch-vocoder iterations. V9 is perceptually rejected and persistent training is "
    "forbidden."
)

NEXT_ARCHITECTURE = "pretrained_vocoder_baseline"
NEXT_GATE = "run_full_utterance_pretrained_vocos_copy_synthesis_before_any_more_vocoder_training"
