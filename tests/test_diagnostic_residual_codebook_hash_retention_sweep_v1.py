from __future__ import annotations

from pathlib import Path

import torch

from scripts.diagnostic_residual_codebook_hash_retention_sweep_v1 import (
    MAX_RETENTION_CAP,
    RETENTION_CAPS,
    _allowed_indices_and_ranks,
    _selected_bucket_records,
)


def _metadata() -> dict[str, object]:
    return {
        "buckets": [
            {
                "voicing_state": "voiced",
                "f0_bin_hz": 100,
                "periodicity_bin": 3,
                "start_index": 0,
                "count": 300,
            },
            {
                "voicing_state": "voiced",
                "f0_bin_hz": 120,
                "periodicity_bin": 3,
                "start_index": 300,
                "count": 700,
            },
            {
                "voicing_state": "unvoiced",
                "f0_bin_hz": -1,
                "periodicity_bin": 1,
                "start_index": 1000,
                "count": 50,
            },
        ]
    }


def test_retention_caps_are_nested_under_one_1024_bank() -> None:
    assert MAX_RETENTION_CAP == 1024
    assert RETENTION_CAPS == (256, 512, 1024)

    selected = _selected_bucket_records(
        _metadata(),
        f0_hz=118.0,
        voiced=1.0,
        periodicity=0.65,
    )
    indices, ranks = _allowed_indices_and_ranks(selected)
    assert indices.shape == ranks.shape
    assert int(indices.numel()) == 1000

    cap128 = indices[ranks < 128]
    cap256 = indices[ranks < 256]
    cap512 = indices[ranks < 512]
    cap1024 = indices[ranks < 1024]

    assert int(cap128.numel()) == 256
    assert int(cap256.numel()) == 512
    assert int(cap512.numel()) == 812
    assert int(cap1024.numel()) == 1000
    assert torch.equal(cap128, cap256[torch.isin(cap256, cap128)])
    assert torch.equal(cap256, cap512[torch.isin(cap512, cap256)])
    assert torch.equal(cap512, cap1024[torch.isin(cap1024, cap512)])


def test_sweep_script_is_diagnostic_only() -> None:
    text = Path("scripts/diagnostic_residual_codebook_hash_retention_sweep_v1.py").read_text(
        encoding="utf-8"
    )
    assert "training_executed\": False" in text
    assert "optimizer_created\": False" in text
    assert "checkpoint_written\": False" in text
    assert "production_codebook_replaced\": False" in text
    assert "torch.optim" not in text
    assert ".backward(" not in text
    assert "cuda" not in text.lower()
