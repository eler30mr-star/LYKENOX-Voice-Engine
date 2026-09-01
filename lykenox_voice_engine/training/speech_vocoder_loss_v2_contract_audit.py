"""Read-only real-data contract audit for owned LYKENOX vocoder loss V2.

The audit proves that the corrected objectives agree with the already validated owned
conditioning contract before any vocoder architecture is selected.  It instantiates no
vocoder model, computes no gradients, performs no parameter update, and writes no checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_loss_v2 import (
    ConditioningAlignedLogMelEnvelopeLossV2,
    OWNED_VOCODER_LOSS_V2_VERSION,
    valid_context_multi_resolution_reconstruction_loss,
)


AUDIT_VERSION = "owned-vocoder-loss-v2-target-consistency-audit-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_ITEMS_PER_SPLIT = 3
SPLITS = ("train", "val")
OUTPUT_DIR_NAME = "owned_vocoder_loss_v2_target_consistency_audit_v1"
TARGET_ZERO_TOLERANCE = 1e-6


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_owned_vocoder_loss_v2_contract_audit(
    root: Path,
    *,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    items_per_split: int = DEFAULT_ITEMS_PER_SPLIT,
    seed: int = 4242,
) -> dict[str, object]:
    root = Path(root).resolve()
    config = LykenoxSpeechConfig()
    envelope_objective = ConditioningAlignedLogMelEnvelopeLossV2(config).cpu()
    protected = _protected(root)
    before = {name: _sha256(path) for name, path in protected.items()}

    envelope_total_values: list[float] = []
    envelope_level_values: list[float] = []
    envelope_slope_values: list[float] = []
    envelope_delta_values: list[float] = []
    reconstruction_total_values: list[float] = []
    reconstruction_logmag_values: list[float] = []
    items: list[dict[str, object]] = []
    exact_frame_contract = True

    with torch.no_grad():
        for split in SPLITS:
            segments, _skipped = collect_owned_vocoder_segments(
                root,
                split,
                segment_mel_frames=segment_mel_frames,
                max_items=items_per_split,
                seed=seed,
            )
            for segment in segments:
                target = segment.waveform.unsqueeze(0)
                conditioning = segment.mel.unsqueeze(0)
                reconstruction = valid_context_multi_resolution_reconstruction_loss(
                    target,
                    target,
                )
                envelope = envelope_objective(target, conditioning)

                frame_contract = (
                    envelope.conditioning_frames == segment_mel_frames
                    and envelope.analysis_frames == segment_mel_frames + 1
                    and envelope.valid_conditioning_frames > 0
                    and envelope.valid_conditioning_frames < segment_mel_frames
                )
                exact_frame_contract = exact_frame_contract and frame_contract

                reconstruction_total = float(reconstruction.total)
                reconstruction_logmag = float(reconstruction.log_magnitude)
                envelope_total = float(envelope.total)
                envelope_level = float(envelope.log_mel_l1)
                envelope_slope = float(envelope.spectral_slope_l1)
                envelope_delta = float(envelope.temporal_delta_l1)

                reconstruction_total_values.append(reconstruction_total)
                reconstruction_logmag_values.append(reconstruction_logmag)
                envelope_total_values.append(envelope_total)
                envelope_level_values.append(envelope_level)
                envelope_slope_values.append(envelope_slope)
                envelope_delta_values.append(envelope_delta)
                items.append(
                    {
                        "split": split,
                        "utterance_id": segment.utterance_id,
                        "start_frame": segment.start_frame,
                        "conditioning_frames": envelope.conditioning_frames,
                        "analysis_frames": envelope.analysis_frames,
                        "valid_conditioning_frames": envelope.valid_conditioning_frames,
                        "reconstruction_valid_frame_counts": list(
                            reconstruction.valid_frame_counts
                        ),
                        "reconstruction_analysis_frame_counts": list(
                            reconstruction.analysis_frame_counts
                        ),
                        "reconstruction_target_self_total": round(reconstruction_total, 10),
                        "reconstruction_target_self_log_magnitude": round(
                            reconstruction_logmag, 10
                        ),
                        "conditioning_aligned_envelope_total": round(envelope_total, 10),
                        "conditioning_aligned_log_mel_l1": round(envelope_level, 10),
                        "conditioning_aligned_spectral_slope_l1": round(
                            envelope_slope, 10
                        ),
                        "conditioning_aligned_temporal_delta_l1": round(
                            envelope_delta, 10
                        ),
                        "frame_contract_exact": frame_contract,
                    }
                )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    target_reconstruction_exact = (
        max(reconstruction_total_values, default=0.0) <= TARGET_ZERO_TOLERANCE
        and max(reconstruction_logmag_values, default=0.0) <= TARGET_ZERO_TOLERANCE
    )
    conditioning_envelope_exact = (
        max(envelope_total_values, default=0.0) <= TARGET_ZERO_TOLERANCE
        and max(envelope_level_values, default=0.0) <= TARGET_ZERO_TOLERANCE
        and max(envelope_slope_values, default=0.0) <= TARGET_ZERO_TOLERANCE
        and max(envelope_delta_values, default=0.0) <= TARGET_ZERO_TOLERANCE
    )
    status_pass = all(
        (
            exact_frame_contract,
            target_reconstruction_exact,
            conditioning_envelope_exact,
            checkpoints_unchanged,
        )
    )

    output_dir = (
        root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    )
    report_path = output_dir / "loss_v2_target_consistency_audit.json"
    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "owned_segment_contract": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "owned_loss_contract": OWNED_VOCODER_LOSS_V2_VERSION,
        "segment_mel_frames": segment_mel_frames,
        "items_per_split": items_per_split,
        "exact_conditioning_frame_contract": exact_frame_contract,
        "target_reconstruction_exact_on_valid_context": target_reconstruction_exact,
        "conditioning_envelope_exact_on_valid_context": conditioning_envelope_exact,
        "mean_reconstruction_target_self_total": round(
            _mean(reconstruction_total_values), 10
        ),
        "mean_reconstruction_target_self_log_magnitude": round(
            _mean(reconstruction_logmag_values), 10
        ),
        "mean_conditioning_aligned_envelope_total": round(
            _mean(envelope_total_values), 10
        ),
        "mean_conditioning_aligned_log_mel_l1": round(
            _mean(envelope_level_values), 10
        ),
        "mean_conditioning_aligned_spectral_slope_l1": round(
            _mean(envelope_slope_values), 10
        ),
        "mean_conditioning_aligned_temporal_delta_l1": round(
            _mean(envelope_delta_values), 10
        ),
        "items": items,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "persistent_training_authorized": False,
        "new_vocoder_architecture_authorized": False,
        "third_party_model_used": False,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "report_path": str(report_path),
        "next_gate": (
            "audit_owned_vocoder_loss_v2_gradient_balance_before_architecture_selection"
            if status_pass
            else "fix_owned_vocoder_loss_v2_target_semantics_before_model_work"
        ),
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_owned_vocoder_loss_v2_contract_audit(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
