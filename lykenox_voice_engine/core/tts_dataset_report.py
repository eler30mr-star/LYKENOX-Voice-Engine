"""Report current neural TTS dataset readiness without training a model."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def build_tts_dataset_report(root: Path, profile: str = "lykenox") -> dict[str, Any]:
    """Build and persist a readiness report for the speech identity dataset."""

    base = root / "datasets" / profile / "identity_voice" / "prepared"
    filtered_report = _read_json(base / "speech" / "quality_report.filtered.json")
    segmented_report = _read_json(base / "speech_segmented" / "quality_report.segmented.json")
    train_csv = base / "speech_segmented" / "train.segmented.csv"
    val_csv = base / "speech_segmented" / "val.segmented.csv"
    report = {
        "profile": profile,
        "purpose": "future neural text-to-speech identity training",
        "status": "dataset_prepared_not_trained",
        "backend_selected": False,
        "model_trained": False,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "train_rows": _count_rows(train_csv),
        "val_rows": _count_rows(val_csv),
        "segmented_minutes": segmented_report.get("minutes", 0),
        "segmented_rows": segmented_report.get("segments", 0),
        "direct_usable_minutes": filtered_report.get("usable_minutes", 0),
        "blocked_rows": filtered_report.get("blocked_rows", 0),
        "blocked_reasons": filtered_report.get("blocked_reasons", {}),
        "ready_for_backend_microtest": bool(segmented_report.get("ready_for_trial_training")),
        "limitations": [
            "No neural backend is installed or selected.",
            "No model checkpoint exists for /speak.",
            "Spanish frontend/backend choice still must be researched before training.",
            "CPU-only training must be proven with a tiny controlled run before use.",
        ],
    }
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "tts_dataset_readiness.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.DictReader(handle)))
