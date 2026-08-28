from __future__ import annotations

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION,
    _run_config,
)


def test_v2_trainer_identity_requires_frame_context() -> None:
    config = LykenoxSpeechConfig(
        vocab_size=30,
        frame_context_version=FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1,
    )
    payload = _run_config(
        config=config,
        train_count=118,
        val_count=14,
        batch_size=2,
        max_epochs=36,
        patience=6,
        seed=1337,
        learning_rate=2e-4,
        weight_decay=1e-4,
        grad_clip=5.0,
        duration_weight=0.10,
        f0_weight=0.25,
        voicing_weight=0.25,
        min_delta=1e-4,
        checkpoint_every_updates=16,
    )
    assert TRAINER_CONTRACT_VERSION == "acoustic-frame-context-bounded-resumable-v2"
    assert FRAME_CONTEXT_VERSION == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
    assert payload["frame_context_version"] == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
    assert payload["frame_context_layers"] == 3
    assert payload["frame_context_kernel_size"] == 5
    assert payload["train_count"] == 118
    assert payload["val_count"] == 14
