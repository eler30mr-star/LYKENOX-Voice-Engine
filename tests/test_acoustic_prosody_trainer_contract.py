from __future__ import annotations

from lykenox_voice_engine.training.speech_acoustic_prosody_train import (
    EpochAccumulator,
    TRAINER_CONTRACT_VERSION,
    TRAIN_ORDER_VERSION,
    _epoch_order,
    _run_config,
)


def test_epoch_order_is_deterministic_and_epoch_specific() -> None:
    first = _epoch_order(20, seed=1337, epoch=1)
    again = _epoch_order(20, seed=1337, epoch=1)
    second = _epoch_order(20, seed=1337, epoch=2)
    assert first == again
    assert first != second
    assert sorted(first) == list(range(20))
    assert sorted(second) == list(range(20))


def test_epoch_accumulator_roundtrip_preserves_partial_epoch() -> None:
    accumulator = EpochAccumulator()
    accumulator.add(
        {
            "total": 1.0,
            "acoustic": 0.7,
            "duration": 0.2,
            "f0": 0.05,
            "voicing": 0.3,
        }
    )
    payload = accumulator.to_payload(epoch=4)
    restored = EpochAccumulator.from_payload(payload, epoch=4)
    assert restored == accumulator
    assert EpochAccumulator.from_payload(payload, epoch=5) == EpochAccumulator()


def test_run_config_is_versioned_and_has_no_execution_budget() -> None:
    config = _run_config(
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
    assert config["trainer_contract_version"] == TRAINER_CONTRACT_VERSION
    assert config["train_order_version"] == TRAIN_ORDER_VERSION
    assert "time_budget_seconds" not in config
    assert "max_updates_this_run" not in config
