"""Compare the legacy pitch-v1 contract with coherent conditioning v2 at known shared anomalies.

No audio generation, training, model inference, checkpoint IO, EQ, denoise, gain change or duration
change occurs. The script reuses the already-localized common anomaly events and reports how the
conditioning semantics change at those exact frames. Policy: LYX-POL-001.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    PITCH_CONDITIONING_V2,
    extract_pitch_conditioning_v2,
)
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import HOP_LENGTH, SAMPLE_RATE


POLICY_ID = "LYX-POL-001"


def _jump_ratio(values, start: int, stop: int) -> float:
    largest = 1.0
    for left, right in zip(range(start, stop - 1), range(start + 1, stop)):
        a = float(values[left])
        b = float(values[right])
        if a <= 0.0 or b <= 0.0:
            continue
        largest = max(largest, max(a, b) / min(a, b))
    return largest


def main() -> None:
    source_path = (
        ROOT
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "generated_vs_reference_diagnostic_v1"
        / "common_conditioning_anomaly_report.json"
    )
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    events = list(source.get("events", []))
    utterances = collect_owned_vocoder_utterances(ROOT, "val", max_items=3)
    by_id = {item.utterance_id: item for item in utterances}
    v2_cache = {}
    rows = []

    for event in events:
        utterance_id = str(event["utterance_id"])
        utterance = by_id.get(utterance_id)
        if utterance is None:
            continue
        if utterance_id not in v2_cache:
            v2_cache[utterance_id] = extract_pitch_conditioning_v2(
                utterance.waveform,
                frame_count=int(utterance.mel_frames),
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                frame_length=int(PITCH_CONFIG["frame_length"]),
                min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
                max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
                anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
                anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
            )
        v2 = v2_cache[utterance_id]
        frame = int(event["analyzed_frame_index"])
        start = max(0, frame - 8)
        stop = min(int(utterance.mel_frames), frame + 9)
        old_f0 = float(utterance.f0_hz[frame])
        old_voiced = float(utterance.voiced[frame])
        old_periodicity = float(utterance.periodicity[frame])
        new_f0 = float(v2.f0_track_hz[frame])
        new_strength = float(v2.periodic_strength[frame])
        energy_confidence = float(v2.energy_confidence[frame])
        anchor = float(v2.anchor_voiced[frame])
        rows.append(
            {
                "utterance_id": utterance_id,
                "time_seconds": float(event["requested_time_seconds"]),
                "shared_candidate_count": int(event["shared_candidate_count"]),
                "tonal_excess_db": float(event["shared_mean_tonal_prominence_excess_db"]),
                "air_excess_db": float(event["shared_mean_air_band_excess_db"]),
                "v1": {
                    "f0_hz": old_f0,
                    "voiced": old_voiced,
                    "raw_periodicity": old_periodicity,
                    "contradictory_unvoiced_periodicity": bool(old_voiced < 0.5 and old_periodicity >= float(PITCH_CONFIG["voiced_periodicity_threshold"])),
                },
                "v2": {
                    "f0_track_hz": new_f0,
                    "periodic_strength": new_strength,
                    "energy_confidence": energy_confidence,
                    "anchor_voiced_for_target_construction_only": anchor,
                    "f0_zero_reset_removed": bool(old_f0 <= 0.0 and new_f0 > 0.0),
                    "periodic_authority_ratio_vs_raw_periodicity": new_strength / max(old_periodicity, 1.0e-8),
                },
                "context": {
                    "v1_nonzero_f0_neighbor_jump_ratio": _jump_ratio(utterance.f0_hz, start, stop),
                    "v2_f0_track_neighbor_jump_ratio": _jump_ratio(v2.f0_track_hz, start, stop),
                    "v1_voiced_state_changes": int(sum(
                        float(utterance.voiced[index]) != float(utterance.voiced[index + 1])
                        for index in range(start, stop - 1)
                    )),
                    "v2_periodic_strength_min": float(v2.periodic_strength[start:stop].min()),
                    "v2_periodic_strength_max": float(v2.periodic_strength[start:stop].max()),
                },
            }
        )

    contradiction_rows = [row for row in rows if row["v1"]["contradictory_unvoiced_periodicity"]]
    zero_reset_rows = [row for row in rows if row["v2"]["f0_zero_reset_removed"]]
    payload = {
        "status": "pitch_conditioning_v2_common_anomaly_audit_complete",
        "policy_id": POLICY_ID,
        "conditioning_version": PITCH_CONDITIONING_V2,
        "audio_generated": False,
        "training_executed": False,
        "model_inference_executed": False,
        "checkpoint_written": False,
        "event_count": len(rows),
        "v1_contradictory_unvoiced_periodicity_event_count": len(contradiction_rows),
        "v2_f0_zero_reset_removed_event_count": len(zero_reset_rows),
        "events": rows,
        "metrics_can_accept_product_quality": False,
        "interpretation_gate": "conditioning_v2_is_semantically_valid_only_if_trusted_anchor_f0_is_preserved_and_transition_contradictions_are_removed_without_claiming_audio_quality",
    }
    output_path = source_path.parent / "pitch_conditioning_v2_common_anomaly_audit.json"
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output_path)

    compact = {
        "status": payload["status"],
        "event_count": len(rows),
        "v1_contradictory_unvoiced_periodicity_event_count": len(contradiction_rows),
        "v2_f0_zero_reset_removed_event_count": len(zero_reset_rows),
        "report": str(output_path),
        "events": [
            {
                "utterance_id": row["utterance_id"],
                "time_seconds": row["time_seconds"],
                "tonal_excess_db": row["tonal_excess_db"],
                "v1_f0_hz": row["v1"]["f0_hz"],
                "v1_voiced": row["v1"]["voiced"],
                "v1_periodicity": row["v1"]["raw_periodicity"],
                "v2_f0_track_hz": row["v2"]["f0_track_hz"],
                "v2_periodic_strength": row["v2"]["periodic_strength"],
                "v2_energy_confidence": row["v2"]["energy_confidence"],
                "v1_jump_ratio": row["context"]["v1_nonzero_f0_neighbor_jump_ratio"],
                "v2_jump_ratio": row["context"]["v2_f0_track_neighbor_jump_ratio"],
            }
            for row in rows
        ],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
