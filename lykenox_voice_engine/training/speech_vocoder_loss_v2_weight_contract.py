"""Frozen owned loss-weight contract for future LYKENOX vocoder work.

This module freezes only the relative authority of already-validated owned objectives.
It does not select or instantiate a vocoder architecture, create an optimizer, authorize
persistent training, modify predicted duration, or permit inference-time post-processing.

The weights were derived from real waveform-gradient norms and then passed a bounded
sensitivity audit over +/-10% relative perturbations.  They are intentionally fixed rather
than recomputed online: adaptive loss reweighting would create a different training contract
and requires a new explicit audit/version.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


OWNED_VOCODER_LOSS_V2_WEIGHT_CONTRACT_VERSION = (
    "owned-vocoder-loss-v2-weight-contract-v1"
)
DATA_CONTRACT_VERSION = "vocoder-segment-v2-full-utterance-mel-pitch-conditioning"
LOSS_CONTRACT_VERSION = "owned-vocoder-loss-v2-valid-context-conditioning-aligned"
PRESENCE_CONTRACT_VERSION = "owned-vocoder-presence-v2-valid-context-target-relative"


@dataclass(frozen=True)
class OwnedVocoderLossV2Weights:
    reconstruction: float = 1.0
    envelope: float = 3.1475
    presence: float = 19.3369
    spectral_balance: float = 60.9496

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction,
            "envelope": self.envelope,
            "presence": self.presence,
            "spectral_balance": self.spectral_balance,
        }


FROZEN_WEIGHTS = OwnedVocoderLossV2Weights()

DERIVATION_RULE = (
    "reconstruction=1; each other weight equals mean reconstruction waveform-gradient "
    "norm divided by that objective's mean waveform-gradient norm"
)
DERIVED_EQUALIZATION_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 3.1475160486,
    "presence": 19.3368866395,
    "spectral_balance": 60.9495610684,
}
CANDIDATE_DERIVATION_RELATIVE_ERRORS = {
    "reconstruction": 0.0,
    "envelope": 0.0000051,
    "presence": 0.00000069,
    "spectral_balance": 0.00000064,
}

SENSITIVITY_AUDIT_VERSION = "owned-vocoder-loss-v2-weight-contract-sensitivity-audit-v1"
SENSITIVITY_AUDIT_STATUS = "pass"
SENSITIVITY_RELATIVE_WEIGHT_PERTURBATION = 0.10
SENSITIVITY_SCENARIO_COUNT = 23
SENSITIVITY_METRICS = {
    "baseline_minimum_weighted_gradient_norm_shares": {
        "reconstruction": 0.16015,
        "envelope": 0.119818,
        "presence": 0.102618,
        "spectral_balance": 0.122951,
    },
    "all_scenarios_minimum_weighted_gradient_norm_shares": {
        "reconstruction": 0.134961,
        "envelope": 0.100216,
        "presence": 0.085556,
        "spectral_balance": 0.102896,
    },
    "baseline_minimum_combined_gradient_alignment_cosines": {
        "reconstruction": 0.34519,
        "envelope": 0.3184,
        "presence": 0.357462,
        "spectral_balance": 0.351217,
    },
    "all_scenarios_minimum_combined_gradient_alignment_cosines": {
        "reconstruction": 0.306252,
        "envelope": 0.271811,
        "presence": 0.310115,
        "spectral_balance": 0.302829,
    },
    "all_scenarios_minimum_first_order_descent_dots": {
        "reconstruction": 57.8851928711,
        "envelope": 10.8785772324,
        "presence": 0.8195143342,
        "spectral_balance": 0.3058912754,
    },
    "baseline_maximum_weighted_gradient_norm_share": 0.526076,
    "all_scenarios_maximum_weighted_gradient_norm_share": 0.575682,
}
SENSITIVITY_GATES = {
    "candidate_tracks_derivation": True,
    "authority_retained": True,
    "alignment_positive": True,
    "alignment_retained": True,
    "descent_positive": True,
    "dominance_bounded": True,
}

# Any of these changes creates a new loss-weight contract version and requires a fresh audit.
ADAPTIVE_LOSS_REWEIGHTING_AUTHORIZED = False
RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED = False
AUTOMATIC_WEIGHT_REDERIVATION_DURING_TRAINING_AUTHORIZED = False

# Freezing this contract does not authorize model work by itself.
MODEL_INSTANTIATION_AUTHORIZED = False
OPTIMIZER_CREATION_AUTHORIZED = False
PERSISTENT_TRAINING_AUTHORIZED = False
NEW_VOCODER_ARCHITECTURE_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False


def combine_owned_vocoder_loss_v2(
    *,
    reconstruction: torch.Tensor,
    envelope: torch.Tensor,
    presence: torch.Tensor,
    spectral_balance: torch.Tensor,
) -> torch.Tensor:
    """Combine the four owned objectives using the frozen V1 weight contract."""

    values = (reconstruction, envelope, presence, spectral_balance)
    if any(value.ndim != 0 for value in values):
        raise ValueError("owned vocoder objective terms must be scalar tensors")
    device = reconstruction.device
    dtype = reconstruction.dtype
    if any(value.device != device for value in values):
        raise ValueError("owned vocoder objective terms must share a device")
    if any(value.dtype != dtype for value in values):
        raise ValueError("owned vocoder objective terms must share a dtype")
    return (
        FROZEN_WEIGHTS.reconstruction * reconstruction
        + FROZEN_WEIGHTS.envelope * envelope
        + FROZEN_WEIGHTS.presence * presence
        + FROZEN_WEIGHTS.spectral_balance * spectral_balance
    )
