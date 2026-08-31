"""Authoritative rejection contract for the LYKENOX V8 vocoder.

V8's fixed STFT/iSTFT renderer was proven numerically sound, but the learned absolute
complex-spectrum predictor introduced strong hop-locked repetition relative to the real
waveform. The corrected v2 smoke measured +0.341784 hop autocorrelation excess and
+0.671333 double-hop excess after only 12 bounded in-memory updates, despite decreasing
complex, magnitude, waveform, and envelope losses. Persistent V8 training is therefore
forbidden; the architecture remains loadable only for forensic reproduction.
"""
from __future__ import annotations


V8_TRAINING_ENABLED = False
V8_ARCHITECTURALLY_REJECTED = True
V8_REJECTION_DATE = "2026-08-31"
V8_REJECTION_GATE = "reference_relative_frame_grid_architecture_smoke_v2"
V8_REJECTION_REASON = (
    "LYKENOX V8 is architecturally rejected: its fixed STFT/iSTFT round-trip is sound, "
    "but learned absolute complex STFT frames add severe hop-locked repetition relative "
    "to real speech (+0.341784 hop and +0.671333 double-hop autocorrelation excess). "
    "Persistent V8 training is disabled; V8 is forensic-only."
)


def require_v8_training_enabled() -> None:
    """Fail before data loading, optimizer construction, or checkpoint writes."""
    if not V8_TRAINING_ENABLED:
        raise RuntimeError(V8_REJECTION_REASON)
