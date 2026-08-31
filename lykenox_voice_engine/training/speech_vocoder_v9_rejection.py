"""Authoritative rejection contract for the LYKENOX V9 vocoder candidate.

V9 fixed the V8 frame-grid failure but its bounded oracle output was perceptually invalid:
the generated audio was described as a short garbage-like sound rather than usable speech.
Metrics cannot override that audible rejection. No exact-resume or persistent V9 training is
allowed; the architecture remains available only for forensic inspection.
"""
from __future__ import annotations


V9_TRAINING_ENABLED = False
V9_PERCEPTUALLY_REJECTED = True
V9_REJECTION_DATE = "2026-08-31"
V9_REJECTION_GATE = "bounded_oracle_listening"
V9_REJECTION_REASON = (
    "LYKENOX V9 is perceptually rejected: bounded oracle output was not recognizable "
    "speech and was heard as a short garbage-like sound. Although V9 removed V8's "
    "reference-relative frame-grid excess and improved several reconstruction metrics, "
    "audible acceptance is authoritative. No exact-resume or persistent training is allowed."
)


def require_v9_training_enabled() -> None:
    if not V9_TRAINING_ENABLED:
        raise RuntimeError(V9_REJECTION_REASON)
