"""First LYKENOX-owned compact acoustic speech model.

This module intentionally implements the model in-project instead of wrapping a
third-party TTS executable. It is an acoustic model bootstrap, not a finished TTS:
waveform generation belongs to a separate LYKENOX vocoder stage.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import LykenoxSpeechConfig


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

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        x = encoded.transpose(1, 2)
        x = self.net[0](x).transpose(1, 2)
        x = self.net[1](x)
        x = self.net[2](x)
        return torch.nn.functional.softplus(self.proj(x).squeeze(-1))


class LykenoxSpeechAcousticModel(nn.Module):
    """Compact duration-conditioned text-to-mel model owned by LYKENOX.

    Inputs:
      token_ids: [batch, text_steps]
      token_mask: optional bool tensor [batch, text_steps], True for valid tokens
      durations: optional integer frame counts [batch, text_steps]

    Outputs:
      mel: [batch, mel_steps, mel_bins]
      duration_prediction: [batch, text_steps]

    During training, ground-truth durations are supplied by the validated LYKENOX
    aligner and must be preserved exactly. ``max_duration_frames`` is only an
    inference safety bound for predicted durations; clipping teacher durations
    would silently shorten the mel target and corrupt supervision.
    """

    def __init__(self, config: LykenoxSpeechConfig) -> None:
        super().__init__()
        self.config = config
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

        padding_mask = None if token_mask is None else ~token_mask.bool()
        encoded = self.encoder(self.embedding(token_ids), src_key_padding_mask=padding_mask)
        duration_prediction = self.duration_predictor(encoded)

        if durations is None:
            regulated_durations = torch.round(duration_prediction).to(torch.long)
            regulated_durations = torch.clamp(
                regulated_durations,
                min=1,
                max=self.config.max_duration_frames,
            )
        else:
            if bool((durations < 0).any().item()):
                raise ValueError("teacher durations must be non-negative")
            regulated_durations = durations.to(torch.long)

        expanded = self._length_regulate(encoded, regulated_durations)
        mel = self.mel_decoder(expanded)
        return {"mel": mel, "duration_prediction": duration_prediction}

    @staticmethod
    def _length_regulate(encoded: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        sequences: list[torch.Tensor] = []
        max_frames = 1
        for batch_index in range(encoded.shape[0]):
            chunks: list[torch.Tensor] = []
            for token_index in range(encoded.shape[1]):
                frames = int(durations[batch_index, token_index].item())
                if frames > 0:
                    chunks.append(encoded[batch_index, token_index].unsqueeze(0).expand(frames, -1))
            sequence = torch.cat(chunks, dim=0) if chunks else encoded.new_zeros((1, encoded.shape[-1]))
            sequences.append(sequence)
            max_frames = max(max_frames, sequence.shape[0])

        output = encoded.new_zeros((encoded.shape[0], max_frames, encoded.shape[-1]))
        for batch_index, sequence in enumerate(sequences):
            output[batch_index, : sequence.shape[0]] = sequence
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
