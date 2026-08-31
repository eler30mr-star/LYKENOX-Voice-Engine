"""Zero-initialized temporal residual postnet for acoustic mel refinement.

The accepted acoustic-v2 model remains immutable. This candidate operates only on its
predicted mel and cannot change duration, F0 or voicing outputs. The final projection is
zero-initialized, so construction preserves the accepted mel exactly before training.
"""
from __future__ import annotations

import torch
from torch import nn


MEL_POSTNET_ARCHITECTURE_V1 = "lykenox-mel-residual-postnet-v1"


class MelResidualPostnetV1(nn.Module):
    """Small temporal residual network over log-mel frames."""

    architecture = MEL_POSTNET_ARCHITECTURE_V1

    def __init__(self, mel_bins: int, hidden_channels: int = 128) -> None:
        super().__init__()
        if mel_bins < 1 or hidden_channels < 1:
            raise ValueError("mel_bins and hidden_channels must be positive")
        self.mel_bins = int(mel_bins)
        self.hidden_channels = int(hidden_channels)
        self.in_conv = nn.Conv1d(mel_bins, hidden_channels, kernel_size=5, padding=2)
        self.mid_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.out_conv = nn.Conv1d(hidden_channels, mel_bins, kernel_size=5, padding=2)
        self.activation = nn.GELU()
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, mel: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, frames, mel_bins]")
        if mel.shape[-1] != self.mel_bins:
            raise ValueError("mel bin count does not match postnet")
        if frame_mask.shape != mel.shape[:2]:
            raise ValueError("frame_mask must match [batch, frames]")
        mask = frame_mask.unsqueeze(-1).to(mel.dtype)
        hidden = (mel * mask).transpose(1, 2)
        hidden = self.activation(self.in_conv(hidden))
        hidden = self.activation(self.mid_conv(hidden))
        residual = self.out_conv(hidden).transpose(1, 2) * mask
        return (mel + residual) * mask


class LykenoxAcousticMelPostnetCandidate(nn.Module):
    """Immutable acoustic-v2 base plus trainable mel-only temporal residual postnet."""

    architecture = MEL_POSTNET_ARCHITECTURE_V1
    duration_training = False
    prosody_training = False
    vocoder_training = False

    def __init__(self, base_model: nn.Module, *, hidden_channels: int = 128) -> None:
        super().__init__()
        self.base_model = base_model
        if not hasattr(base_model, "config"):
            raise ValueError("base_model must expose speech config")
        self.config = base_model.config
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.postnet = MelResidualPostnetV1(
            int(self.config.mel_bins),
            hidden_channels=hidden_channels,
        )

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.base_model(token_ids, token_mask, durations)
        refined_mel = self.postnet(base["mel"], base["mel_mask"])
        output = dict(base)
        output["base_mel"] = base["mel"]
        output["mel"] = refined_mel
        return output
