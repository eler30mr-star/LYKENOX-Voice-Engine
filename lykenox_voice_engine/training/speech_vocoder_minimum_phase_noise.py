"""Deterministic per-example aperiodic-noise seed derivation for the owned renderer."""

from __future__ import annotations

import hashlib


NOISE_SEED_VERSION = "owned-minimum-phase-per-example-noise-seed-v1"


def stable_owned_noise_seed(
    base_seed: int,
    *,
    split: str,
    utterance_id: str,
    start_frame: int = 0,
) -> int:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    payload = f"{int(base_seed)}:{split}:{utterance_id}:{int(start_frame)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Keep the seed in a conservative signed-31-bit range for reproducible tensor arithmetic.
    return int.from_bytes(digest[:8], "big", signed=False) % 2147483647


__all__ = ["NOISE_SEED_VERSION", "stable_owned_noise_seed"]
