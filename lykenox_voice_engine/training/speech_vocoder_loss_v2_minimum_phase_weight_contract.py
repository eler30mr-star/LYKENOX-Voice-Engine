"""Historical equal-norm Loss V2 weights for the owned minimum-phase vocoder.

Waveform-space weight contract v1 remains valid evidence in its original calibration space.
This v2 candidate then equalized mean gradient norms in cepstrum space, but the real-data
integrated preflight proved that equal norm is not enough: several objectives remained
anti-aligned with the combined direction in cepstrum and predictor parameter space.

Therefore these weights are retained only for forensic reproducibility and are rejected for
training.  The active path uses deterministic pre-training directional calibration to choose
one fixed common-descent vector, then freezes it for the whole run and resume.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


OWNED_MINIMUM_PHASE_LOSS_WEIGHT_CONTRACT_VERSION = (
    "owned-vocoder-loss-v2-minimum-phase-weight-contract-v2"
)
SOURCE_WAVEFORM_WEIGHT_CONTRACT_VERSION = "owned-vocoder-loss-v2-weight-contract-v1"
ARCHITECTURE_FAMILY = "owned_minimum_phase_time_varying_filter_over_neutral_excitation"
DERIVATION_SPACE = "cepstrum_space"
CROSS_CHECK_SPACE = "parameter_space"

V1_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 3.1475,
    "presence": 19.3369,
    "spectral_balance": 60.9496,
}
MEASURED_CEPSTRUM_NEUTRAL_V1_SHARES = {
    "reconstruction": 0.04117481310162209,
    "envelope": 0.008032201568982572,
    "presence": 0.1949435194527072,
    "spectral_balance": 0.7558494658766881,
}
MEASURED_PARAMETER_NEUTRAL_V1_SHARES = {
    "reconstruction": 0.046928433266268124,
    "envelope": 0.009529865430589276,
    "presence": 0.234379802425534,
    "spectral_balance": 0.7091618988776086,
}

DERIVATION_RULE = (
    "v2_reconstruction=1; v2_i=v1_i*(cepstrum_v1_reconstruction_share/"
    "cepstrum_v1_objective_share)"
)


@dataclass(frozen=True)
class OwnedMinimumPhaseLossV2Weights:
    reconstruction: float = 1.0
    envelope: float = 16.1348
    presence: float = 4.0842
    spectral_balance: float = 3.3202

    def as_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction,
            "envelope": self.envelope,
            "presence": self.presence,
            "spectral_balance": self.spectral_balance,
        }


FROZEN_MINIMUM_PHASE_WEIGHTS = OwnedMinimumPhaseLossV2Weights()
DERIVED_CEPSTRUM_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 16.13476991635949,
    "presence": 4.084225244830007,
    "spectral_balance": 3.32022248068645,
}
DERIVED_PARAMETER_CROSS_CHECK_WEIGHTS = {
    "reconstruction": 1.0,
    "envelope": 15.499404979157768,
    "presence": 3.8717091312286214,
    "spectral_balance": 4.033309235497123,
}
PARAMETER_VS_CEPSTRUM_RELATIVE_DIFFERENCE = {
    "reconstruction": 0.0,
    "envelope": 0.03937861776123062,
    "presence": 0.05203339699993233,
    "spectral_balance": 0.21477077483772827,
}
PROJECTED_CEPSTRUM_MEAN_WEIGHTED_SHARES = {
    "reconstruction": 0.25,
    "envelope": 0.25,
    "presence": 0.25,
    "spectral_balance": 0.25,
}
PROJECTED_PARAMETER_MEAN_WEIGHTED_SHARES = {
    "reconstruction": 0.25516173577957985,
    "envelope": 0.26562154507212243,
    "presence": 0.2691674316078851,
    "spectral_balance": 0.21004928754041263,
}

# Real integrated preflight evidence rejecting this candidate for training.
DIRECTIONAL_PREFLIGHT_STATUS = "fail"
DIRECTIONAL_REJECTION_REASON = (
    "equalized_norms_but_negative_common_direction_alignments_and_descent_dots_remained"
)
DIRECTIONAL_FAILURE_EVIDENCE = {
    "cepstrum_neutral_spectral_balance_alignment": -0.0285,
    "cepstrum_neutral_spectral_balance_descent_dot": -591.84,
    "cepstrum_connected_presence_alignment": -0.0108,
    "cepstrum_connected_spectral_balance_alignment": -0.0134,
    "cepstrum_connected_presence_descent_dot": -131647.89,
    "cepstrum_connected_spectral_balance_descent_dot": -51244.09,
    "parameter_neutral_envelope_alignment": -0.0519,
    "parameter_neutral_envelope_descent_dot": -1580.42,
    "parameter_connected_presence_alignment": -0.0216,
    "parameter_connected_presence_descent_dot": -1879.15,
}

ADAPTIVE_LOSS_REWEIGHTING_AUTHORIZED = False
RUNTIME_WEIGHT_OVERRIDE_AUTHORIZED = False
AUTOMATIC_WEIGHT_REDERIVATION_DURING_TRAINING_AUTHORIZED = False
THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
MINIMUM_PHASE_WEIGHT_V2_IMPLEMENTED = True
MINIMUM_PHASE_WEIGHT_V2_DIRECTIONALLY_COMPATIBLE = False
MINIMUM_PHASE_WEIGHT_V2_IS_ACTIVE_CANDIDATE = False
PERSISTENT_TRAINING_AUTHORIZED = False


def combine_owned_minimum_phase_loss_v2(
    *,
    reconstruction: torch.Tensor,
    envelope: torch.Tensor,
    presence: torch.Tensor,
    spectral_balance: torch.Tensor,
) -> torch.Tensor:
    """Historical v2 combination retained for forensic reproducibility only."""

    values = (reconstruction, envelope, presence, spectral_balance)
    if any(value.ndim != 0 for value in values):
        raise ValueError("owned minimum-phase objective terms must be scalar tensors")
    device = reconstruction.device
    dtype = reconstruction.dtype
    if any(value.device != device for value in values):
        raise ValueError("owned minimum-phase objective terms must share a device")
    if any(value.dtype != dtype for value in values):
        raise ValueError("owned minimum-phase objective terms must share a dtype")
    weights = FROZEN_MINIMUM_PHASE_WEIGHTS
    return (
        weights.reconstruction * reconstruction
        + weights.envelope * envelope
        + weights.presence * presence
        + weights.spectral_balance * spectral_balance
    )
