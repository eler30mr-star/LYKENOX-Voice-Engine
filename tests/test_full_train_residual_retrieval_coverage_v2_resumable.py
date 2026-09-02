from __future__ import annotations

from pathlib import Path

from scripts.diagnostic_full_train_residual_retrieval_coverage_v2_resumable import (
    ROW_FIELDS,
    _append_checkpoint_row,
    _load_checkpoint_rows,
)


def _row(index: int) -> dict[str, object]:
    return {
        "utterance_id": "heldout_demo",
        "vector_index": index,
        "conditioning_index": index,
        "f0_hz": 120.0,
        "voiced": 1.0,
        "periodicity": 0.9,
        "sampled_compatible_candidate_count": 128,
        "full_train_compatible_candidate_count": 4096,
        "sampled_best_filtered_response_cosine": 0.80,
        "sampled_best_normalized_mse": 0.36,
        "full_train_best_filtered_response_cosine": 0.90,
        "full_train_best_normalized_mse": 0.19,
        "normalized_mse_improvement_full_vs_sampled": 0.17,
    }


def test_checkpoint_roundtrip_preserves_completed_window_indices(tmp_path: Path) -> None:
    path = tmp_path / "progress.csv"
    _append_checkpoint_row(path, _row(3))
    _append_checkpoint_row(path, _row(7))

    loaded = _load_checkpoint_rows(path, utterance_id="heldout_demo")

    assert sorted(loaded) == [3, 7]
    assert loaded[3]["full_train_best_normalized_mse"] == 0.19
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == ROW_FIELDS


def test_checkpoint_duplicate_window_uses_latest_completed_record(tmp_path: Path) -> None:
    path = tmp_path / "progress.csv"
    first = _row(2)
    second = _row(2)
    second["full_train_best_normalized_mse"] = 0.11
    _append_checkpoint_row(path, first)
    _append_checkpoint_row(path, second)

    loaded = _load_checkpoint_rows(path, utterance_id="heldout_demo")

    assert list(loaded) == [2]
    assert loaded[2]["full_train_best_normalized_mse"] == 0.11
