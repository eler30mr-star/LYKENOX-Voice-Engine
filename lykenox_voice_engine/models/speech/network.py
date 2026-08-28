"""First LYKENOX-owned compact acoustic speech model.

This module intentionally implements the model in-project instead of wrapping a
third-party TTS executable. It is an acoustic model bootstrap, not a finished TTS:
waveform generation belongs to a separate LYKENOX vocoder stage.
"""

from __future__ import annotations

import math

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
        # Transformer padding positions can contain residual activations even when
        # excluded as attention keys. Zero them before the temporal convolution so
        # a padded neighbor cannot affect the final valid token in a shorter item.
        masked_encoded = encoded * valid.unsqueeze(-1).to(encoded.dtype)
        x = masked_encoded.transpose(1, 2)
        x = self.net[0](x).transpose(1, 2)
        x = self.net[1](x)
        x = self.net[2](x)
        prediction = torch.nn.functional.softplus(self.proj(x).squeeze(-1))
        return torch.where(valid, prediction, torch.zeros_like(prediction))


class FrameProsodyPredictor(nn.Module):
    """Predict frame-level log-F0 and voicing from regulated acoustic features.

    The F0 projection is initialized to 100 Hz in log space. This is only a stable
    optimization prior; supervised pitch targets determine the learned contour.
    Voicing is returned as logits so training can use masked BCEWithLogitsLoss.
    """

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


class LykenoxSpeechAcousticModel(nn.Module):
    """Compact duration-conditioned text-to-acoustics model owned by LYKENOX.

    Inputs:
      token_ids: [batch, text_steps]
      token_mask: optional bool tensor [batch, text_steps], True for valid tokens
      durations: optional integer frame counts [batch, text_steps]

    Outputs:
      mel: [batch, mel_steps, mel_bins]
      f0_prediction_hz: [batch, mel_steps], positive F0 hypothesis per frame
      f0_log_prediction: [batch, mel_steps], log-Hz representation used for training
      voicing_logits: [batch, mel_steps], binary voiced/unvoiced logits
      duration_prediction: [batch, text_steps]
      mel_mask: [batch, mel_steps], True for real regulated frames
      mel_lengths: [batch], exact regulated frame count per item

    During training, ground-truth durations are supplied by the validated LYKENOX
    aligner and must be preserved exactly. ``max_duration_frames`` is only an
    inference safety bound for predicted durations; clipping teacher durations
    would silently shorten the mel target and corrupt supervision.

    F0/voicing heads operate after length regulation, so their frame grid is exactly
    the same grid consumed by the accepted LYKENOX v4.1 source-filter vocoder.

    The length regulator is tensorized: it contains no Python loop over batch or
    token positions and no duration-dependent ``.item()`` calls. This keeps the
    core path suitable for later dynamic ONNX/export work and makes padded batch
    behavior explicit through ``mel_mask`` and ``mel_lengths``.
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
            regulated_durations = torch.round(duration_prediction).to(torch.long)
            regulated_durations = torch.clamp(
                regulated_durations,
                min=1,
                max=self.config.max_duration_frames,
            )
            regulated_durations = torch.where(
                valid_tokens,
                regulated_durations,
                torch.zeros_like(regulated_durations),
            )
        else:
            # Training/teacher durations are validated at the aligned-dataset
            # boundary. Never clamp real timing supervision inside the model.
            regulated_durations = torch.where(
                valid_tokens,
                durations.to(torch.long),
                torch.zeros_like(durations, dtype=torch.long),
            )

        expanded, mel_mask, mel_lengths = self._length_regulate(
            encoded,
            regulated_durations,
        )
        mel = self.mel_decoder(expanded)
        f0_log_prediction, voicing_logits = self.frame_prosody_predictor(expanded)
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
            "mel_mask": mel_mask,
            "mel_lengths": mel_lengths,
        }

    @staticmethod
    def _length_regulate(
        encoded: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand encoded tokens into frames with tensor-only indexing.

        ``durations`` may contain zero-duration structural/padded tokens. For each
        frame position, the first cumulative token boundary strictly greater than
        that position determines the owning token. The implementation is batch
        vectorized and therefore does not bake Python-side sequence lengths into
        the graph.
        """

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
        cumulative_ends = torch.cumsum(durations, dim=1)

        # Count how many token ends are <= each frame position. Zero-duration
        # tokens naturally disappear because their cumulative boundary repeats.
        token_indices = torch.sum(
            frame_positions.view(1, -1, 1)
            >= cumulative_ends.unsqueeze(1),
            dim=-1,
        )
        token_indices = torch.clamp(token_indices, max=encoded.shape[1] - 1)
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