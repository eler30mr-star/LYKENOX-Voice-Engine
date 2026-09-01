"""Active objective for the owned minimum-phase vocoder.

This is the only loss assembly authorized for the minimum-phase architecture.  The generic
waveform-space weight contract v1 is historical evidence only and must not be imported by
minimum-phase training code.  This module combines the already-validated Loss V2 objective
terms with the architecture-coupled v2 weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    OWNED_VOCODER_LOSS_V2_VERSION,
    valid_context_multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2_minimum_phase_weight_contract import (
    FROZEN_MINIMUM_PHASE_WEIGHTS,
    OWNED_MINIMUM_PHASE_LOSS_WEIGHT_CONTRACT_VERSION,
    combine_owned_minimum_phase_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import (
    OWNED_VOCODER_PRESENCE_V2_VERSION,
    target_relative_presence_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION = "owned-minimum-phase-objective-v2"
ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION = OWNED_MINIMUM_PHASE_LOSS_WEIGHT_CONTRACT_VERSION
HISTORICAL_WAVEFORM_WEIGHT_CONTRACT_AUTHORIZED = False
ADAPTIVE_REWEIGHTING_AUTHORIZED = False
RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED = False


@dataclass(frozen=True)
class MinimumPhaseObjectiveResult:
    total: torch.Tensor
    reconstruction: torch.Tensor
    envelope: torch.Tensor
    presence: torch.Tensor
    spectral_balance: torch.Tensor

    def detached_terms(self) -> dict[str, float]:
        return {
            "reconstruction": float(self.reconstruction.detach()),
            "envelope": float(self.envelope.detach()),
            "presence": float(self.presence.detach()),
            "spectral_balance": float(self.spectral_balance.detach()),
        }


class OwnedMinimumPhaseObjectiveV2(torch.nn.Module):
    """Compute the four owned objectives and combine them with the minimum-phase v2 weights."""

    def __init__(self, config: LykenoxSpeechConfig | None = None) -> None:
        super().__init__()
        self.config = config or LykenoxSpeechConfig()
        self.envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(self.config)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        conditioning_log_mel: torch.Tensor,
    ) -> MinimumPhaseObjectiveResult:
        reconstruction = valid_context_multi_resolution_reconstruction_loss(
            prediction,
            target,
        ).total
        envelope = self.envelope_objective(prediction, conditioning_log_mel).total
        presence = target_relative_presence_loss_v2(
            prediction,
            target,
            sample_rate=self.config.sample_rate,
        ).loss
        spectral_balance = target_relative_spectral_balance_loss(
            prediction,
            target,
            sample_rate=self.config.sample_rate,
        ).loss
        total = combine_owned_minimum_phase_loss_v2(
            reconstruction=reconstruction,
            envelope=envelope,
            presence=presence,
            spectral_balance=spectral_balance,
        )
        return MinimumPhaseObjectiveResult(
            total=total,
            reconstruction=reconstruction,
            envelope=envelope,
            presence=presence,
            spectral_balance=spectral_balance,
        )


def active_weights() -> dict[str, float]:
    return FROZEN_MINIMUM_PHASE_WEIGHTS.as_dict()


__all__ = [
    "ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION",
    "ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION",
    "HISTORICAL_WAVEFORM_WEIGHT_CONTRACT_AUTHORIZED",
    "ADAPTIVE_REWEIGHTING_AUTHORIZED",
    "RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED",
    "OWNED_VOCODER_LOSS_V2_VERSION",
    "OWNED_VOCODER_PRESENCE_V2_VERSION",
    "MinimumPhaseObjectiveResult",
    "OwnedMinimumPhaseObjectiveV2",
    "active_weights",
]
