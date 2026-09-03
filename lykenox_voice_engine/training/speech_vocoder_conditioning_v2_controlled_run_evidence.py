"""Recorded evidence for the one authorized Continuous Source V2 + pitch-conditioning-v2 retrain.

This is engineering evidence only. Metrics may reject/localize but cannot accept product quality.
Policy: LYX-POL-001.
"""

POLICY_ID = "LYX-POL-001"
EVIDENCE_VERSION = "owned-vocoder-conditioning-v2-controlled-run-evidence-v1"

STATUS = "training_complete_awaiting_direct_reference_and_human_listening"
CONTROLLED_CHANGE = "pitch_conditioning_v1_to_v2_only"
ARCHITECTURE_CHANGED = False
LOSS_FUNCTION_CHANGED = False
RENDERER_CHANGED = False
STEP3F_TARGET_CHANGED = False

ARCHITECTURE = "lykenox_owned_continuous_residual_source_v2_level_factored"
CONDITIONING_CONTRACT = "lykenox-pitch-conditioning-v2-continuous-strength"
UPDATES = 600
BEST_VAL_TOTAL = 2.899257183074951
BEST_CHECKPOINT = "models/lykenox_identity/training/continuous_residual_source_v2_pitch_conditioning_v2/best.pt"

CODEBOOK_USED = False
POSTHOC_GAIN_NORMALIZATION_USED = False
POSTHOC_EQ_USED = False
POSTHOC_DENOISING_USED = False
PRODUCTION_ACCEPTED_BY_METRICS = False

HELDOUT_CANDIDATE_REFERENCE_RMS_RATIOS = {
    "speech_0021": 0.8470,
    "speech_0022": 0.8729,
    "speech_0024": 0.9176,
}
HELDOUT_HISTORICAL_V2_REFERENCE_RMS_RATIOS = {
    "speech_0021": 0.8331,
    "speech_0022": 0.9048,
    "speech_0024": 0.9493,
}

LEVEL_INTERPRETATION = (
    "candidate level remains broadly comparable to historical V2; no amplitude collapse occurred, "
    "but level does not establish perceptual improvement"
)

# This run was authorized only to test the verified transition-conditioning correction. It may not be
# used to claim the separate stable-voiced tonal defect at speech_0024 ~4.00 s is solved without direct
# evidence. The next gate is direct reference comparison of the new WAVs followed by complete human
# listening against the historical V2 baseline and identity ceiling.
TRANSITION_CONDITIONING_HYPOTHESIS_ACCEPTED_BY_METRICS = False
STABLE_VOICED_TONAL_DEFECT_RESOLVED = False
FURTHER_RETRAIN_FROM_THIS_HYPOTHESIS_AUTHORIZED = False

NEXT_ACTION = (
    "rerun direct generated-vs-reference diagnostic so the new controlled candidate is compared "
    "against historical V2 at the already-localized anomaly times, then perform complete listening"
)
