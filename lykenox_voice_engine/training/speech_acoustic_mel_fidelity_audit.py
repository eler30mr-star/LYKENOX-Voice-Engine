"""Read-only held-out mel fidelity audit for accepted acoustic frame-context v2.

The v4.2 level-attribution gate localizes the main clarity/presence loss to predicted mel
conditioning while teacher durations are held fixed.  This audit stays entirely upstream
of the vocoder and measures whether the accepted acoustic v2 mel is spectrally or temporally
smoothed relative to the cached mel-v1 target.  It does not train, change durations, apply
gain/EQ, or mutate checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torchaudio

from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)


AUDIT_VERSION = "acoustic-mel-fidelity-heldout-audit-v1"
OUTPUT_DIR_NAME = "acoustic_mel_fidelity_audit_v1"
BANDS_HZ = (
    ("80_300", 80.0, 300.0),
    ("300_1000", 300.0, 1000.0),
    ("1k_3k", 1000.0, 3000.0),
    ("3k_8k", 3000.0, 8000.0),
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _db_ratio(value: float, reference: float) -> float:
    if value <= 0.0 or reference <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(value / reference)


def _mel_bin_centers_hz(
    *,
    sample_rate: int,
    n_fft: int,
    mel_bins: int,
) -> torch.Tensor:
    """Return the FFT frequency at the peak of each torchaudio HTK mel filter."""
    fbanks = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=float(sample_rate) / 2.0,
        n_mels=mel_bins,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )
    fft_hz = torch.linspace(0.0, float(sample_rate) / 2.0, n_fft // 2 + 1)
    return fft_hz[torch.argmax(fbanks, dim=0)].to(torch.float64)


def _band_masks(centers_hz: torch.Tensor) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    for name, low, high in BANDS_HZ:
        mask = (centers_hz >= low) & (centers_hz < high)
        if not bool(mask.any()):
            raise RuntimeError(f"Mel band {name} contains no filter centers")
        masks[name] = mask
    return masks


def _mel_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    centers_hz: torch.Tensor,
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must share [frames, mel_bins] shape")
    if prediction.shape[1] != centers_hz.numel():
        raise ValueError("mel-bin centers do not match mel shape")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("mel tensors must be finite")

    pred = prediction.to(torch.float64)
    ref = target.to(torch.float64)
    mel_l1 = float(torch.mean(torch.abs(pred - ref)))

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    ref_centered = ref - ref.mean(dim=1, keepdim=True)
    centered_shape_l1 = float(torch.mean(torch.abs(pred_centered - ref_centered)))

    pred_spectral_delta = float(torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1])))
    ref_spectral_delta = float(torch.mean(torch.abs(ref[:, 1:] - ref[:, :-1])))
    if pred.shape[0] > 1:
        pred_temporal_delta = float(torch.mean(torch.abs(pred[1:] - pred[:-1])))
        ref_temporal_delta = float(torch.mean(torch.abs(ref[1:] - ref[:-1])))
    else:
        pred_temporal_delta = ref_temporal_delta = 0.0

    pred_linear = torch.exp(pred)
    ref_linear = torch.exp(ref)
    result: dict[str, float] = {
        "mel_l1": mel_l1,
        "centered_shape_l1": centered_shape_l1,
        "prediction_spectral_delta_l1": pred_spectral_delta,
        "target_spectral_delta_l1": ref_spectral_delta,
        "spectral_delta_ratio": pred_spectral_delta / max(ref_spectral_delta, 1e-12),
        "prediction_temporal_delta_l1": pred_temporal_delta,
        "target_temporal_delta_l1": ref_temporal_delta,
        "temporal_delta_ratio": pred_temporal_delta / max(ref_temporal_delta, 1e-12),
    }
    for name, mask in _band_masks(centers_hz).items():
        pred_mean = float(pred_linear[:, mask].mean())
        ref_mean = float(ref_linear[:, mask].mean())
        result[f"band_{name}_relative_db"] = _db_ratio(pred_mean, ref_mean)
    return result


def run_acoustic_mel_fidelity_audit(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    training_dir = root / "models" / "lykenox_identity" / "training"
    checkpoint = training_dir / "acoustic_frame_context_v2" / "best.pt"
    last_checkpoint = training_dir / "acoustic_frame_context_v2" / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Accepted acoustic v2 best checkpoint not found: {checkpoint}")

    protected = {"best": checkpoint, "last": last_checkpoint}
    before = {name: _sha256(path) for name, path in protected.items()}

    model, payload = load_acoustic_prosody_checkpoint(checkpoint)
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("Accepted acoustic v2 checkpoint is missing run_config")
    identity_exact = (
        model.config.frame_context_version == FRAME_CONTEXT_VERSION
        and run_config.get("trainer_contract_version") == TRAINER_CONTRACT_VERSION
        and run_config.get("frame_context_version") == FRAME_CONTEXT_VERSION
    )
    if not identity_exact:
        raise RuntimeError("Mel fidelity audit requires accepted acoustic frame-context v2")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        model.config,
        duration_root=duration_root,
        include_pitch_targets=False,
    )
    if len(dataset) < 1:
        raise RuntimeError("Validation dataset is empty")

    centers_hz = _mel_bin_centers_hz(
        sample_rate=model.config.sample_rate,
        n_fft=model.config.n_fft,
        mel_bins=model.config.mel_bins,
    )
    masks = _band_masks(centers_hz)
    model.cpu().eval()

    aggregate: dict[str, list[float]] = {}
    per_utterance: list[dict[str, object]] = []
    frame_contracts_exact = True
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            batch = collate_aligned_speech([item]).to("cpu")
            output = model(batch.token_ids, batch.token_mask, batch.durations)
            exact = (
                output["mel"].shape == batch.mel.shape
                and torch.equal(output["mel_mask"], batch.mel_mask)
                and torch.equal(output["mel_lengths"], batch.mel_lengths)
                and torch.equal(output["regulated_durations"], batch.durations)
            )
            frame_contracts_exact = frame_contracts_exact and exact
            if not exact:
                raise RuntimeError(f"Teacher-grid mel contract failed for {item['utterance_id']}")

            frames = int(batch.mel_lengths[0])
            metrics = _mel_metrics(
                output["mel"][0, :frames],
                batch.mel[0, :frames],
                centers_hz=centers_hz,
            )
            for name, value in metrics.items():
                aggregate.setdefault(name, []).append(float(value))
            per_utterance.append(
                {
                    "dataset_index": index,
                    "utterance_id": str(item["utterance_id"]),
                    "text": str(item["text"]),
                    "mel_frames": frames,
                    **{name: round(float(value), 6) for name, value in metrics.items()},
                }
            )

    means = {
        name: round(sum(values) / len(values), 6)
        for name, values in aggregate.items()
    }
    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    structural_gate_pass = identity_exact and frame_contracts_exact and checkpoints_unchanged

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / OUTPUT_DIR_NAME
    )
    report_path = output_dir / "mel_fidelity_report.json"
    report: dict[str, object] = {
        "status": "needs_review" if structural_gate_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "acoustic_identity_exact": identity_exact,
        "teacher_duration_grid_used": True,
        "predicted_duration_modified": False,
        "all_frame_contracts_exact": frame_contracts_exact,
        "checkpoints_unchanged": checkpoints_unchanged,
        "training_started": False,
        "training_authorized": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "validation_item_count": len(per_utterance),
        "mel_bin_center_hz": [round(float(value), 3) for value in centers_hz.tolist()],
        "band_bin_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "mean_metrics": means,
        "items": per_utterance,
        "interpretation": (
            "Negative target-relative dB in 1-3 kHz or 3-8 kHz identifies acoustic-mel "
            "underpresence before the vocoder. Spectral/temporal delta ratios materially below "
            "1.0 identify smoothing. This audit is diagnostic only and does not authorize training."
        ),
        "next_gate": "decide_minimal_acoustic_mel_loss_change_from_fidelity_metrics",
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_acoustic_mel_fidelity_audit(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
