"""Authoritative rejection contract for the LYKENOX V6 vocoder.

V6 is retained only so historical checkpoints and diagnostics remain reproducible.
Persistent or resumed V6 training is prohibited after the 2026-08-30 full-utterance
oracle listening test produced unintelligible, nasal/gangoso, periodic output that was
materially worse than the accepted v4.2 baseline.
"""

from __future__ import annotations


V6_TRAINING_ENABLED = False
V6_PERCEPTUALLY_REJECTED = True
V6_REJECTION_DATE = "2026-08-30"
V6_REJECTION_GATE = "full_utterance_oracle_listening"
V6_REJECTION_REASON = (
    "LYKENOX V6 is perceptually rejected: all three full-utterance oracle outputs "
    "were materially less intelligible than v4.2 and exhibited stronger periodic "
    "whine plus nasal/gangoso coloration. Training is disabled; V6 checkpoints are "
    "for forensic inspection only."
)


def require_v6_training_enabled() -> None:
    """Fail before data loading, checkpoint writes, or optimizer construction."""
    if not V6_TRAINING_ENABLED:
        raise RuntimeError(V6_REJECTION_REASON)
