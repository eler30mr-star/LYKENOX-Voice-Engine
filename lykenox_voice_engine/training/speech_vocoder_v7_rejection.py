"""Authoritative rejection contract for the LYKENOX V7 vocoder.

V7 is retained only for forensic reproducibility. Full-utterance oracle listening on
2026-08-30 showed that all three outputs collapsed to a nearly input-independent frame-grid
tone instead of intelligible speech. The dominant lines occur exactly at 93.75 Hz and
187.5 Hz, matching ``sample_rate / hop_length`` and its second harmonic for 24 kHz / 256.
This is consistent with transposed-convolution checkerboard/grid leakage.
"""

from __future__ import annotations


V7_TRAINING_ENABLED = False
V7_PERCEPTUALLY_REJECTED = True
V7_REJECTION_DATE = "2026-08-30"
V7_REJECTION_GATE = "full_utterance_oracle_listening"
V7_REJECTION_REASON = (
    "LYKENOX V7 is perceptually rejected: the epoch-1 full-utterance oracle produced "
    "nearly pure frame-grid tones instead of intelligible speech. All three outputs "
    "showed dominant 93.75 Hz / 187.5 Hz lines and approximately 0.99 normalized "
    "autocorrelation at the 256-sample hop. Training is disabled; V7 checkpoints are "
    "for forensic inspection only."
)


def require_v7_training_enabled() -> None:
    """Fail before data loading, optimizer construction, or checkpoint writes."""
    if not V7_TRAINING_ENABLED:
        raise RuntimeError(V7_REJECTION_REASON)
