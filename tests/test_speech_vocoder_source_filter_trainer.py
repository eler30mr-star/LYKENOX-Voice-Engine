from __future__ import annotations

from lykenox_voice_engine.training.speech_vocoder_source_filter_train import (
    EpochAccumulator,
    TRAINER_CONTRACT_VERSION,
    _run_config,
)


def _config() -> dict[str, object]:
    return _run_config(
        segment_mel_frames=64,
        train_items=118,
        val_items=14,
        max_epochs=24,
        warmup_epochs=8,
        patience=6,
        seed=1337,
        validation_seed=101340,
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        balance_weight=0.5,
        adversarial_weight=0.05,
        feature_matching_weight=1.0,
        min_delta=1e-4,
        checkpoint_every_updates=16,
        val_segment_set_sha256="abc",
    )


def test_trainer_run_config_is_stable_training_identity() -> None:
    config = _config()
    assert config["trainer_contract_version"] == TRAINER_CONTRACT_VERSION
    assert "time_budget_seconds" not in config
    assert "max_updates_this_run" not in config
    assert config["validation_segment_set_sha256"] == "abc"


def test_epoch_accumulator_roundtrip_preserves_partial_epoch_metrics() -> None:
    accumulator = EpochAccumulator(
        reconstruction_sum=3.5,
        balance_sum=1.25,
        update_count=4,
        discriminator_sum=2.0,
        adversarial_sum=0.5,
        feature_matching_sum=0.75,
        adversarial_count=2,
    )
    payload = accumulator.to_payload(epoch=7)
    restored = EpochAccumulator.from_payload(payload, epoch=7)
    assert restored == accumulator


def test_epoch_accumulator_rejects_other_epoch_partial_state() -> None:
    payload = EpochAccumulator(update_count=4).to_payload(epoch=7)
    restored = EpochAccumulator.from_payload(payload, epoch=8)
    assert restored == EpochAccumulator()
