"""Scoped authorization for the owned minimum-phase train-and-listen candidate.

This contract implements the explicit training gate required by LYX-POL-001 without opening
general persistent vocoder training.  The only authorized learned path is the LYKENOX-owned
frame-rate cepstral predictor, trained on owned V2 conditioning with the active minimum-phase
Loss V2 weight contract.  A read-only V2 authority preflight must pass before the optimizer
is created.  The run is CPU-only, update-bounded, exactly resumable, and may create only its
own candidate checkpoints and complete held-out listening artifacts.
"""

from __future__ import annotations


CONTRACT_VERSION = "owned-minimum-phase-bounded-train-and-listen-contract-v1"
POLICY_ID = "LYX-POL-001"
POLICY_VERSION = "1.0"

CPU_ONLY = True
TRAIN_AND_LISTEN_AUTHORIZED = True
MAX_UPDATES_AUTHORIZED = 400
V2_AUTHORITY_PREFLIGHT_REQUIRED = True
EXACT_RESUME_REQUIRED = True
OWNED_V2_DATA_REQUIRED = True
ACTIVE_MINIMUM_PHASE_OBJECTIVE_V2_REQUIRED = True
SCOPED_CHECKPOINT_CREATION_AUTHORIZED = True
COMPLETE_HELDOUT_AUDIO_REQUIRED = True
HUMAN_LISTENING_REQUIRED_FOR_ACCEPTANCE = True

GENERAL_PERSISTENT_TRAINING_AUTHORIZED = False
GENERAL_OPTIMIZER_CREATION_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
METRICS_CAN_ACCEPT_VOICE_QUALITY = False


def require_authorized_run(max_updates: int) -> None:
    if not TRAIN_AND_LISTEN_AUTHORIZED:
        raise RuntimeError("minimum-phase bounded train-and-listen route is not authorized")
    if max_updates < 1 or max_updates > MAX_UPDATES_AUTHORIZED:
        raise RuntimeError(
            f"minimum-phase candidate is authorized for 1..{MAX_UPDATES_AUTHORIZED} updates"
        )


__all__ = [
    "CONTRACT_VERSION",
    "POLICY_ID",
    "POLICY_VERSION",
    "CPU_ONLY",
    "TRAIN_AND_LISTEN_AUTHORIZED",
    "MAX_UPDATES_AUTHORIZED",
    "V2_AUTHORITY_PREFLIGHT_REQUIRED",
    "EXACT_RESUME_REQUIRED",
    "OWNED_V2_DATA_REQUIRED",
    "ACTIVE_MINIMUM_PHASE_OBJECTIVE_V2_REQUIRED",
    "SCOPED_CHECKPOINT_CREATION_AUTHORIZED",
    "COMPLETE_HELDOUT_AUDIO_REQUIRED",
    "HUMAN_LISTENING_REQUIRED_FOR_ACCEPTANCE",
    "GENERAL_PERSISTENT_TRAINING_AUTHORIZED",
    "GENERAL_OPTIMIZER_CREATION_AUTHORIZED",
    "THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED",
    "PREDICTED_DURATION_MODIFICATION_AUTHORIZED",
    "POSTHOC_GAIN_NORMALIZATION_AUTHORIZED",
    "POSTHOC_EQ_AUTHORIZED",
    "POSTHOC_DENOISING_AUTHORIZED",
    "METRICS_CAN_ACCEPT_VOICE_QUALITY",
    "require_authorized_run",
]
