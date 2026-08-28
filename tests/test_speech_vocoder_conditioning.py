from __future__ import annotations

import torch

from lykenox_voice_engine.runtime.speech_conditioning import (
    PREDICTED_SPEECH_F0_MAX_HZ,
    PREDICTED_SPEECH_F0_MIN_HZ,
    SPEECH_VOCODER_CONDITIONING_VERSION,
    prepare_speech_vocoder_conditioning,
)


def test_reference_free_conditioning_zeros_unvoiced_and_padding() -> None:
    output = {
        "mel": torch.ones((1, 4, 80), dtype=torch.float32),
        "f0_prediction_hz": torch.tensor([[40.0, 100.0, 500.0, 120.0]]),
        "voicing_logits": torch.tensor([[10.0, -10.0, 10.0, 10.0]]),
        "mel_mask": torch.tensor([[True, True, True, False]]),
    }
    conditioning = prepare_speech_vocoder_conditioning(output)

    assert SPEECH_VOCODER_CONDITIONING_VERSION == "speech-vocoder-conditioning-v1"
    assert conditioning.f0_hz[0, 0].item() == PREDICTED_SPEECH_F0_MIN_HZ
    assert conditioning.f0_hz[0, 1].item() == 0.0
    assert conditioning.f0_hz[0, 2].item() == PREDICTED_SPEECH_F0_MAX_HZ
    assert conditioning.f0_hz[0, 3].item() == 0.0
    assert conditioning.voiced.tolist() == [[1.0, 0.0, 1.0, 0.0]]
    assert conditioning.mel[0, 3].abs().sum().item() == 0.0
    assert conditioning.raw_f0_hz[0, 3].item() == 0.0
    assert conditioning.f0_clipped_mask.tolist() == [[True, False, True, False]]


def test_reference_free_conditioning_requires_frame_exact_shapes() -> None:
    output = {
        "mel": torch.zeros((1, 3, 80)),
        "f0_prediction_hz": torch.zeros((1, 2)),
        "voicing_logits": torch.zeros((1, 3)),
        "mel_mask": torch.ones((1, 3), dtype=torch.bool),
    }
    try:
        prepare_speech_vocoder_conditioning(output)
    except ValueError as exc:
        assert "match mel" in str(exc)
    else:
        raise AssertionError("shape mismatch must be rejected")
