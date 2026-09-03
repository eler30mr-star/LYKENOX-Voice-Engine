"""Run the direct WAV/reference diagnostic and print a compact cross-variant locator summary.

No audio is generated. No model/checkpoint is loaded. This wrapper exists so one command is enough to
compare all already-generated held-out speech against reference and identify anomaly times shared by
multiple source architectures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from diagnose_vocoder_generated_vs_reference_v1 import diagnose


TIME_BIN_SECONDS = 0.05


def _time_bin(value: float) -> float:
    return round(round(float(value) / TIME_BIN_SECONDS) * TIME_BIN_SECONDS, 3)


def _common_anomalies(comparisons: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_utterance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_utterance[str(item["utterance_id"])].append(item)

    result: dict[str, list[dict[str, Any]]] = {}
    for utterance_id, items in by_utterance.items():
        buckets: dict[float, dict[str, Any]] = {}
        metric_lists = {
            "locator_score": "locator_scores",
            "log_spectral_mae_db": "log_spectral_mae_db",
            "rms_delta_db": "rms_delta_db",
            "tonal_prominence_excess_db": "tonal_prominence_excess_db",
            "high_band_excess_db": "high_band_excess_db",
            "air_band_excess_db": "air_band_excess_db",
            "high_band_flatness_delta": "high_band_flatness_delta",
        }
        for item in items:
            candidate_key = str(item["candidate_key"])
            # One vote per candidate per time bin prevents a single candidate's adjacent top frames
            # from dominating the cross-variant consensus.
            seen: set[float] = set()
            for anomaly in item.get("top_anomaly_timestamps", []):
                bin_time = _time_bin(float(anomaly["time_seconds"]))
                if bin_time in seen:
                    continue
                seen.add(bin_time)
                bucket = buckets.setdefault(
                    bin_time,
                    {
                        "time_seconds": bin_time,
                        "candidate_keys": set(),
                        "locator_scores": [],
                        "log_spectral_mae_db": [],
                        "rms_delta_db": [],
                        "tonal_prominence_excess_db": [],
                        "high_band_excess_db": [],
                        "air_band_excess_db": [],
                        "high_band_flatness_delta": [],
                    },
                )
                bucket["candidate_keys"].add(candidate_key)
                for source_key, list_key in metric_lists.items():
                    bucket[list_key].append(float(anomaly[source_key]))

        rows: list[dict[str, Any]] = []
        for bucket in buckets.values():
            candidate_keys = sorted(bucket["candidate_keys"])
            count = len(candidate_keys)
            if count < 2:
                continue

            def mean(key: str) -> float:
                values = bucket[key]
                return sum(values) / max(len(values), 1)

            rows.append(
                {
                    "time_seconds": bucket["time_seconds"],
                    "candidate_count": count,
                    "candidate_keys": candidate_keys,
                    "mean_locator_score": mean("locator_scores"),
                    "mean_log_spectral_mae_db": mean("log_spectral_mae_db"),
                    "mean_rms_delta_db": mean("rms_delta_db"),
                    "mean_tonal_prominence_excess_db": mean("tonal_prominence_excess_db"),
                    "mean_high_band_excess_db": mean("high_band_excess_db"),
                    "mean_air_band_excess_db": mean("air_band_excess_db"),
                    "mean_high_band_flatness_delta": mean("high_band_flatness_delta"),
                }
            )
        rows.sort(key=lambda row: (-int(row["candidate_count"]), -float(row["mean_locator_score"])))
        result[utterance_id] = rows[:12]
    return result


def run(root: Path) -> dict[str, object]:
    compact = diagnose(root)
    report_path = Path(str(compact["json_report"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    comparisons = list(report["comparisons"])
    common = _common_anomalies(comparisons)

    unified = [
        item
        for item in comparisons
        if "unified_phase_residual_source_v1" in str(item.get("candidate", ""))
    ]
    unified_summary = [
        {
            "utterance_id": item["utterance_id"],
            "rms_ratio": item["rms_ratio"],
            "rms_delta_db": item["rms_delta_db"],
            "log_spectral_mae_db": item["log_spectral_mae_db"],
            "log_spectral_mae_excess_over_ceiling_db": item.get("excess_over_ceiling", {}).get("log_spectral_mae_db"),
            "body_band_delta_db": item["band_energy_delta_db"]["body"],
            "presence_band_delta_db": item["band_energy_delta_db"]["presence"],
            "high_band_delta_db": item["band_energy_delta_db"]["high"],
            "air_band_delta_db": item["band_energy_delta_db"]["air"],
            "tonal_prominence_excess_db_p95": item["tonal_prominence_excess_db_p95"],
            "high_band_flatness_delta_p95": item["high_band_flatness_delta_p95"],
            "terminal_regions": item["terminal_regions"],
            "top_anomaly_timestamps": item["top_anomaly_timestamps"][:6],
        }
        for item in unified
    ]

    return {
        "status": "direct_reference_comparison_complete",
        "audio_generated": False,
        "training_executed": False,
        "model_inference_executed": False,
        "metrics_can_accept_product_quality": False,
        "comparison_count": compact["comparison_count"],
        "json_report": compact["json_report"],
        "csv_summary": compact["csv_summary"],
        "closest_candidates_diagnostic_only": compact["top_rankings"],
        "unified_source_direct_reference_comparison": unified_summary,
        "cross_variant_common_anomaly_times": common,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
