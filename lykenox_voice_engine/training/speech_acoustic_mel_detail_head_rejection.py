"""Permanent perceptual rejection record for the frame-hidden mel detail-head candidate."""

REJECTION_VERSION = "acoustic-frame-hidden-mel-detail-perceptual-rejection-v1"
CANDIDATE = "lykenox-frame-hidden-mel-detail-head-v1"
PERCEPTUAL_VERDICT = "effectively_equal_to_base_no_audible_progress"
EPOCH2_TRAINING_AUTHORIZED = False
PERSISTENT_TRAINING_COMPLETE = False
ACCEPTED_BASELINE = "vocoder-v4.2-intelligible-but-colored-baseline"
NEXT_GATE = "review_reference_vs_v4_2_forensics_before_any_more_training"

REJECTION_RATIONALE = (
    "Full held-out A/B was perceptually equal to the base route and therefore failed the "
    "required clear-audible-improvement gate. No epoch 2 is authorized. Subsequent work "
    "moves to direct original-waveform versus v4.2-oracle forensics because coloration "
    "remains even when target acoustic/prosody conditioning is used."
)
