from __future__ import annotations

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.models.speech.config import (
    FRAME_CONTEXT_NONE,
    FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1,
)


def test_historical_default_keeps_frame_context_disabled() -> None:
    config = LykenoxSpeechConfig(vocab_size=30)
    assert config.frame_context_version == FRAME_CONTEXT_NONE
    model = LykenoxSpeechAcousticModel(config)
    assert model.frame_context is None
    assert not any(key.startswith("frame_context.") for key in model.state_dict())


def test_position_features_vary_inside_one_token() -> None:
    durations = torch.tensor([[0, 5, 0]], dtype=torch.long)
    encoded = torch.zeros((1, 3, 8), dtype=torch.float32)
    expanded, mel_mask, mel_lengths = LykenoxSpeechAcousticModel._length_regulate(
        encoded,
        durations,
    )
    features = LykenoxSpeechAcousticModel._regulated_position_features(
        durations,
        mel_mask,
        mel_lengths,
    )
    assert expanded.shape[1] == 5
    assert features.shape == (1, 5, 3)
    assert torch.all(features[0, 1:, 0] > features[0, :-1, 0])
    assert torch.allclose(features[0, :, 1], features[0, :1, 1].expand(5))
    assert torch.all(features[0, 1:, 2] > features[0, :-1, 2])


def test_contextual_model_can_break_repeated_token_symmetry() -> None:
    torch.manual_seed(7)
    config = LykenoxSpeechConfig(
        vocab_size=30,
        hidden_size=32,
        encoder_layers=1,
        encoder_heads=4,
        ff_multiplier=2,
        mel_bins=8,
        frame_context_version=FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1,
        frame_context_layers=2,
        frame_context_kernel_size=3,
    )
    model = LykenoxSpeechAcousticModel(config).eval()
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    durations = torch.tensor([[0, 6, 0]], dtype=torch.long)
    with torch.no_grad():
        output = model(token_ids, token_mask, durations)
    mel = output["mel"][0, :6]
    assert not torch.allclose(mel[1:], mel[:-1])
    assert output["mel_lengths"].tolist() == [6]


def test_zero_duration_tokens_remain_supported_with_context() -> None:
    config = LykenoxSpeechConfig(
        vocab_size=30,
        hidden_size=32,
        encoder_layers=1,
        encoder_heads=4,
        ff_multiplier=2,
        mel_bins=8,
        frame_context_version=FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1,
        frame_context_layers=1,
        frame_context_kernel_size=3,
    )
    model = LykenoxSpeechAcousticModel(config).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    durations = torch.tensor([[0, 2, 0, 3]], dtype=torch.long)
    with torch.no_grad():
        output = model(token_ids, token_mask, durations)
    assert output["mel_lengths"].tolist() == [5]
    assert bool(output["mel_mask"].all())
