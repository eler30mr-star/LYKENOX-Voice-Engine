from __future__ import annotations

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_losses import (
    masked_log_f0_loss,
    masked_voicing_loss,
)


def _config() -> LykenoxSpeechConfig:
    return LykenoxSpeechConfig(
        vocab_size=16,
        hidden_size=32,
        encoder_layers=1,
        encoder_heads=4,
        ff_multiplier=2,
        dropout=0.0,
        mel_bins=8,
    )


def test_acoustic_model_emits_frame_aligned_prosody_heads() -> None:
    model = LykenoxSpeechAcousticModel(_config()).eval()
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    token_mask = torch.tensor([[True, True, True], [True, True, False]])
    durations = torch.tensor([[2, 1, 2], [1, 3, 0]], dtype=torch.long)

    with torch.no_grad():
        output = model(token_ids, token_mask, durations)

    assert output["mel"].shape == (2, 5, 8)
    assert output["f0_prediction_hz"].shape == (2, 5)
    assert output["f0_log_prediction"].shape == (2, 5)
    assert output["voicing_logits"].shape == (2, 5)
    assert output["mel_lengths"].tolist() == [5, 4]
    assert torch.all(output["f0_prediction_hz"][output["mel_mask"]] > 0.0)
    assert torch.all(output["f0_prediction_hz"][~output["mel_mask"]] == 0.0)
    assert torch.all(output["f0_log_prediction"][~output["mel_mask"]] == 0.0)
    assert torch.all(output["voicing_logits"][~output["mel_mask"]] == 0.0)


def test_f0_loss_uses_only_target_voiced_real_frames() -> None:
    prediction = torch.tensor([[100.0, 120.0, 999.0, 999.0]], requires_grad=True)
    target = torch.tensor([[100.0, 100.0, 0.0, 0.0]])
    voiced = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    loss = masked_log_f0_loss(prediction, target, voiced, mask)
    loss.backward()

    assert prediction.grad is not None
    assert float(prediction.grad[0, 0]) == 0.0
    assert float(prediction.grad[0, 2]) == 0.0
    assert float(prediction.grad[0, 3]) == 0.0
    assert float(prediction.grad[0, 1]) != 0.0


def test_voicing_loss_ignores_padded_frames() -> None:
    logits_a = torch.tensor([[0.0, 0.0, 100.0]])
    logits_b = torch.tensor([[0.0, 0.0, -100.0]])
    target = torch.tensor([[1.0, 0.0, 1.0]])
    mask = torch.tensor([[True, True, False]])

    loss_a = masked_voicing_loss(logits_a, target, mask)
    loss_b = masked_voicing_loss(logits_b, target, mask)
    assert torch.equal(loss_a, loss_b)
