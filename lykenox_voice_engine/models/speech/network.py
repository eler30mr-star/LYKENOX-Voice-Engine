"""First LYKENOX-owned compact acoustic speech model.

This module intentionally implements the model in-project instead of wrapping a
third-party TTS executable. It is an acoustic model bootstrap, not a finished TTS:
waveform generation belongs to a separate LYKENOX vocoder stage.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import (
    FRAME_CONTEXT_NONE,
    FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1,
    LykenoxSpeechConfig,
)
from .duration_policy import regulate_predicted_durations


class DurationPredictor(nn.Module):
    """Predict non-negative frame durations per input token."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        self.proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoded: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = (
            torch.ones(encoded.shape[:2], dtype=torch.bool, device=encoded.device)
            if token_mask is None
            else token_mask.bool()
        )
        masked_encoded = encoded * valid.unsqueeze(-1).to(encoded.dtype)
        x = masked_encoded.transpose(1, 2)
        x = self.net[0](x).transpose(1, 2)
        x = self.net[1](x)
        x = self.net[2](x)
        prediction = torch.nn.functional.softplus(self.proj(x).squeeze(-1))
        return torch.where(valid, prediction, torch.zeros_like(prediction))


class FrameProsodyPredictor(nn.Module):
    """Predict frame-level log-F0 and voicing from regulated acoustic features."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.log_f0_proj = nn.Linear(hidden_size, 1)
        self.voicing_proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.log_f0_proj.weight)
        nn.init.constant_(self.log_f0_proj.bias, math.log(100.0))

    def forward(self, expanded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(expanded)
        log_f0 = self.log_f0_proj(hidden).squeeze(-1)
        voicing_logits = self.voicing_proj(hidden).squeeze(-1)
        return log_f0, voicing_logits


class DepthwiseFrameContextBlock(nn.Module):
    """CPU-compact residual temporal context over regulated acoustic frames."""

    def __init__(self, hidden_size: int, *, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("frame context kernel_size must be odd and >= 3")
        if dilation < 1:
            raise ValueError("frame context dilation must be positive")
        padding = dilation * (kernel_size // 2)
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_size,
        )
        self.pointwise = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        mask = frame_mask.unsqueeze(-1).to(x.dtype)
        residual = x
        hidden = self.norm(x) * mask
        hidden = self.depthwise(hidden.transpose(1, 2))
        hidden = self.activation(hidden)
        hidden = self.pointwise(hidden).transpose(1, 2)
        return (residual + hidden) * mask


class PostRegulationFrameContext(nn.Module):
    """Inject explicit intra-token position plus local temporal context.

    Length regulation repeats one token representation for every acoustic frame owned by
    that token. Without an additional frame coordinate, a feed-forward decoder is exactly
    constant inside the token. This module adds three deterministic frame coordinates:

    - centered progress inside the owning token;
    - log duration of the owning token;
    - normalized progress through the full utterance.

    Residual depthwise-separable convolutions then model coarticulation across neighboring
    frames while remaining small enough for the CPU-only product target.
    """

    FEATURE_COUNT = 3

    def __init__(
        self,
        hidden_size: int,
        *,
        layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("frame context layers must be positive")
        self.position_projection = nn.Sequential(
            nn.Linear(self.FEATURE_COUNT, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DepthwiseFrameContextBlock(
                    hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                )
                for layer in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        expanded: torch.Tensor,
        position_features: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        if position_features.shape != (*expanded.shape[:2], self.FEATURE_COUNT):
            raise ValueError("position_features do not match regulated frame shape")
        mask = frame_mask.unsqueeze(-1).to(expanded.dtype)
        hidden = expanded + self.position_projection(position_features.to(expanded.dtype))
        hidden = hidden * mask
        for block in self.blocks:
            hidden = block(hidden, frame_mask)
        return self.final_norm(hidden) * mask


class LykenoxSpeechAcousticModel(nn.Module):
    """Compact duration-conditioned text-to-acoustics model owned by LYKENOX.

    Historical checkpoints use ``frame_context_version='none'`` and preserve the original
    piecewise-constant decoder exactly. New training can select
    ``token-progress-conv-v1`` to make mel/F0/voicing explicitly frame-expressive after
    length regulation while retaining the same text, duration and vocoder contracts.

    Teacher durations are always preserved exactly. When no durations are supplied,
    inference uses the separately versioned token-aware predicted-duration policy instead
    of the historical fixed 1..80 clamp.
    """

    def __init__(self, config: LykenoxSpeechConfig) -> None:
        super().__init__()
        self.config = config
        if config.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.encoder_heads,
            dim_feedforward=config.hidden_size * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.duration_predictor = DurationPredictor(config.hidden_size)
        self.mel_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.mel_bins),
        )
        self.frame_prosody_predictor = FrameProsodyPredictor(config.hidden_size)

        if config.frame_context_version == FRAME_CONTEXT_NONE:
            self.frame_context: PostRegulationFrameContext | None = None
        elif config.frame_context_version == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1:
            self.frame_context = PostRegulationFrameContext(
                config.hidden_size,
                layers=config.frame_context_layers,
                kernel_size=config.frame_context_kernel_size,
            )
        else:
            raise ValueError(
                f"Unsupported frame_context_version: {config.frame_context_version}"
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, text_steps]")
        if token_mask is not None and token_mask.shape != token_ids.shape:
            raise ValueError("token_mask must match token_ids shape")
        if durations is not None and durations.shape != token_ids.shape:
            raise ValueError("durations must match token_ids shape")

        valid_tokens = (
            torch.ones_like(token_ids, dtype=torch.bool)
            if token_mask is None
            else token_mask.bool()
        )
        encoded = self.encoder(
            self.embedding(token_ids),
            src_key_padding_mask=~valid_tokens,
        )
        duration_prediction = self.duration_predictor(encoded, valid_tokens)

        if durations is None:
            regulated_durations = regulate_predicted_durations(
                token_ids,
                valid_tokens,
                duration_prediction,
            )
        else:
            regulated_durations = torch.where(
                valid_tokens,
                durations.to(torch.long),
                torch.zeros_like(durations, dtype=torch.long),
            )

        expanded, mel_mask, mel_lengths = self._length_regulate(
            encoded,
            regulated_durations,
        )
        frame_hidden = expanded
        if self.frame_context is not None:
            position_features = self._regulated_position_features(
                regulated_durations,
                mel_mask,
                mel_lengths,
            )
            frame_hidden = self.frame_context(
                expanded,
                position_features,
                mel_mask,
            )

        mel = self.mel_decoder(frame_hidden)
        f0_log_prediction, voicing_logits = self.frame_prosody_predictor(frame_hidden)
        f0_prediction_hz = torch.exp(f0_log_prediction)

        frame_mask = mel_mask.to(mel.dtype)
        mel = mel * frame_mask.unsqueeze(-1)
        f0_prediction_hz = f0_prediction_hz * frame_mask
        f0_log_prediction = f0_log_prediction * frame_mask
        voicing_logits = voicing_logits * frame_mask
        return {
            "mel": mel,
            "f0_prediction_hz": f0_prediction_hz,
            "f0_log_prediction": f0_log_prediction,
            "voicing_logits": voicing_logits,
            "duration_prediction": duration_prediction,
            "regulated_durations": regulated_durations,
            "mel_mask": mel_mask,
            "mel_lengths": mel_lengths,
        }

    @staticmethod
    def _regulated_token_indices(
        durations: torch.Tensor,
        frame_count: torch.Tensor | int,
    ) -> torch.Tensor:
        frame_positions = torch.arange(
            frame_count,
            device=durations.device,
            dtype=torch.long,
        )
        cumulative_ends = torch.cumsum(durations.to(torch.long), dim=1)
        token_indices = torch.sum(
            frame_positions.view(1, -1, 1)
            >= cumulative_ends.unsqueeze(1),
            dim=-1,
        )
        return torch.clamp(token_indices, max=durations.shape[1] - 1)

    @classmethod
    def _regulated_position_features(
        cls,
        durations: torch.Tensor,
        mel_mask: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Create tensor-only frame coordinates aligned to the length regulator."""

        if durations.ndim != 2:
            raise ValueError("durations must have shape [batch, text_steps]")
        if mel_mask.ndim != 2 or mel_mask.shape[0] != durations.shape[0]:
            raise ValueError("mel_mask batch dimension must match durations")
        if mel_lengths.ndim != 1 or mel_lengths.shape[0] != durations.shape[0]:
            raise ValueError("mel_lengths must have shape [batch]")

        frame_count = mel_mask.shape[1]
        token_indices = cls._regulated_token_indices(durations, frame_count)
        durations_long = durations.to(torch.long)
        cumulative_ends = torch.cumsum(durations_long, dim=1)
        cumulative_starts = cumulative_ends - durations_long
        owning_duration = torch.gather(durations_long, 1, token_indices).clamp_min(1)
        owning_start = torch.gather(cumulative_starts, 1, token_indices)

        frame_positions = torch.arange(
            frame_count,
            device=durations.device,
            dtype=torch.long,
        ).unsqueeze(0)
        offset = frame_positions - owning_start
        token_progress = (
            (offset.to(torch.float32) + 0.5) / owning_duration.to(torch.float32)
        )
        centered_token_progress = token_progress * 2.0 - 1.0
        duration_feature = torch.log1p(owning_duration.to(torch.float32)) / math.log(128.0)
        utterance_progress = (
            (frame_positions.to(torch.float32) + 0.5)
            / mel_lengths.clamp_min(1).unsqueeze(1).to(torch.float32)
        )
        features = torch.stack(
            (centered_token_progress, duration_feature, utterance_progress),
            dim=-1,
        )
        return features * mel_mask.unsqueeze(-1).to(features.dtype)

    @classmethod
    def _length_regulate(
        cls,
        encoded: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand encoded tokens into frames with tensor-only indexing."""

        if encoded.ndim != 3:
            raise ValueError("encoded must have shape [batch, text_steps, hidden]")
        if durations.ndim != 2 or durations.shape[:2] != encoded.shape[:2]:
            raise ValueError("durations must have shape [batch, text_steps]")

        durations = durations.to(torch.long)
        mel_lengths = durations.sum(dim=1)
        max_frames = torch.clamp(mel_lengths.max(), min=1)
        frame_positions = torch.arange(
            max_frames,
            device=encoded.device,
            dtype=torch.long,
        )
        token_indices = cls._regulated_token_indices(durations, max_frames)
        gather_index = token_indices.unsqueeze(-1).expand(
            -1,
            -1,
            encoded.shape[-1],
        )
        expanded = torch.gather(encoded, dim=1, index=gather_index)
        mel_mask = frame_positions.unsqueeze(0) < mel_lengths.unsqueeze(1)
        expanded = expanded * mel_mask.unsqueeze(-1).to(expanded.dtype)
        return expanded, mel_mask, mel_lengths

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
