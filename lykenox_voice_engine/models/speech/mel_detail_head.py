"""Zero-init mel detail head fed from acoustic frame_hidden before the mel bottleneck.

Unlike the rejected mel postnet, this candidate does not try to reconstruct missing detail
from an already-smoothed predicted mel. It captures the immutable acoustic-v2 frame_hidden
input to ``mel_decoder`` and learns a temporal residual mel projection from that richer
representation. Duration, F0, voicing, encoder, frame context and the accepted mel decoder
remain frozen.
"""
from __future__ import annotations

import torch
from torch import nn


MEL_DETAIL_HEAD_ARCHITECTURE_V1 = "lykenox-frame-hidden-mel-detail-head-v1"


class FrameHiddenMelDetailHeadV1(nn.Module):
    architecture = MEL_DETAIL_HEAD_ARCHITECTURE_V1

    def __init__(self, hidden_size: int, mel_bins: int) -> None:
        super().__init__()
        if hidden_size < 1 or mel_bins < 1:
            raise ValueError("hidden_size and mel_bins must be positive")
        self.hidden_size = int(hidden_size)
        self.mel_bins = int(mel_bins)
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise_1 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size
        )
        self.pointwise_1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.depthwise_2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=5,
            padding=4,
            dilation=2,
            groups=hidden_size,
        )
        self.pointwise_2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.out_proj = nn.Conv1d(hidden_size, mel_bins, kernel_size=1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, frame_hidden: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if frame_hidden.ndim != 3:
            raise ValueError("frame_hidden must have shape [batch, frames, hidden]")
        if frame_hidden.shape[-1] != self.hidden_size:
            raise ValueError("frame_hidden hidden size mismatch")
        if frame_mask.shape != frame_hidden.shape[:2]:
            raise ValueError("frame_mask must match [batch, frames]")
        mask = frame_mask.unsqueeze(-1).to(frame_hidden.dtype)
        hidden = self.norm(frame_hidden) * mask
        hidden = self.depthwise_1(hidden.transpose(1, 2))
        hidden = self.activation(self.pointwise_1(hidden))
        hidden = self.depthwise_2(hidden)
        hidden = self.activation(self.pointwise_2(hidden))
        return self.out_proj(hidden).transpose(1, 2) * mask


class LykenoxAcousticFrameHiddenDetailCandidate(nn.Module):
    """Immutable acoustic-v2 base plus trainable frame-hidden mel residual head."""

    architecture = MEL_DETAIL_HEAD_ARCHITECTURE_V1
    duration_training = False
    prosody_training = False
    vocoder_training = False

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(base_model, "config") or not hasattr(base_model, "mel_decoder"):
            raise ValueError("base_model must expose acoustic config and mel_decoder")
        self.base_model = base_model
        self.config = base_model.config
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.detail_head = FrameHiddenMelDetailHeadV1(
            int(self.config.hidden_size),
            int(self.config.mel_bins),
        )

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        captured: list[torch.Tensor] = []

        def capture_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if not args:
                raise RuntimeError("mel_decoder pre-hook did not receive frame_hidden")
            captured.append(args[0].detach())

        handle = self.base_model.mel_decoder.register_forward_pre_hook(capture_input)
        try:
            with torch.no_grad():
                base = self.base_model(token_ids, token_mask, durations)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("failed to capture exactly one frame_hidden tensor")
        frame_hidden = captured[0]
        residual = self.detail_head(frame_hidden, base["mel_mask"])
        output = dict(base)
        output["base_mel"] = base["mel"]
        output["frame_hidden_for_detail"] = frame_hidden
        output["mel"] = base["mel"] + residual
        return output
