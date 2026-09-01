"""Fixed directional Loss V2 objective for the owned minimum-phase vocoder.

Weights are calibrated once before model optimizer creation, then frozen into the run config
and checkpoint.  No adaptive/runtime reweighting is permitted after training starts.
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
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_directional_weight_calibration import (
    CALIBRATION_VERSION,
    DirectionalFixedWeights,
)
from lykenox_voice_engine.training.speech_vocoder_presence_v2 import (
    OWNED_VOCODER_PRESENCE_V2_VERSION,
    target_relative_presence_loss_v2,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import (
    target_relative_spectral_balance_loss,
)


ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION = "owned-minimum-phase-objective-v3-directional-fixed"
ADAPTIVE_REWEIGHTING_AUTHORIZED = False
RUNTIME_WEIGHT_OVERRIDE_AFTER_START_AUTHORIZED = False
AUTOMATIC_WEIGHT_REDERIVATION_DURING_TRAINING_AUTHORIZED = False


@dataclass(frozen=True)
class MinimumPhaseObjectiveResultV3:
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


class OwnedMinimumPhaseObjectiveV3(torch.nn.Module):
    def __init__(
        self,
        weights: DirectionalFixedWeights,
        config: LykenoxSpeechConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LykenoxSpeechConfig()
        self.weights = weights
        self.envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(self.config)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        conditioning_log_mel: torch.Tensor,
    ) -> MinimumPhaseObjectiveResultV3:
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
        total = (
            self.weights.reconstruction * reconstruction
            + self.weights.envelope * envelope
            + self.weights.presence * presence
            + self.weights.spectral_balance * spectral_balance
        )
        return MinimumPhaseObjectiveResultV3(
            total=total,
            reconstruction=reconstruction,
            envelope=envelope,
            presence=presence,
            spectral_balance=spectral_balance,
        )


__all__ = [
    "ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION",
    "CALIBRATION_VERSION",
    "OWNED_VOCODER_LOSS_V2_VERSION",
    "OWNED_VOCODER_PRESENCE_V2_VERSION",
    "OwnedMinimumPhaseObjectiveV3",
    "MinimumPhaseObjectiveResultV3",
]
