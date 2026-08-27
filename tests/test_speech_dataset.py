from __future__ import annotations

import torch

from lykenox_voice_engine.training.speech_dataset import uniform_bootstrap_durations


def test_uniform_bootstrap_durations_preserve_frame_count() -> None:
    durations = uniform_bootstrap_durations(7, 31)
    assert durations.dtype == torch.long
    assert durations.shape == (7,)
    assert int(durations.sum()) == 31
    assert int(durations.max() - durations.min()) <= 1


def test_uniform_bootstrap_durations_reject_invalid_values() -> None:
    for token_count, frames in ((0, 10), (3, 0), (-1, 5)):
        try:
            uniform_bootstrap_durations(token_count, frames)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
