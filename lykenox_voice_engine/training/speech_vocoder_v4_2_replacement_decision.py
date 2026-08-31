"""Frozen engineering decision after direct v4.2 oracle/source-filter forensics."""

DECISION_VERSION = "vocoder-v4-2-replacement-decision-v2"
V4_2_ROLE = "intelligible_colored_baseline_only"
V4_2_FURTHER_TRAINING_AUTHORIZED = False
ACOUSTIC_TRAINING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_EQ_DENOISE_AUTHORIZED = False

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

NEXT_ARCHITECTURE = "lykenox_phase_increment_spectral_ola_v9"
NEXT_GATE = "run_v9_phase_increment_architecture_smoke_before_any_persistent_training"
