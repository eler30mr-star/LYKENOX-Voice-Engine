"""Read-only forensics for crop-edge semantics in LYKENOX-owned vocoder losses.

Historical vocoder losses analyze waveform crops with centered STFT/mel transforms. A
centered transform reflects samples at the crop boundary, even though the real utterance
contains valid context outside that crop. This audit measures that discrepancy directly by
comparing crop-local analysis against the corresponding frames from full-utterance analysis.

No vocoder model is instantiated, no gradients are computed, and no checkpoint is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torchaudio

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import LykenoxVocoderConfig
from lykenox_voice_engine.training.speech_aligner_train import _dataset
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    _mono_waveform,
    collect_owned_vocoder_segments,
)
from lykenox_voice_engine.training.speech_vocoder_losses import STFT_RESOLUTIONS


AUDIT_VERSION = "owned-vocoder-loss-edge-semantics-forensics-v1"
DEFAULT_SEGMENT_MEL_FRAMES = 64
DEFAULT_ITEMS_PER_SPLIT = 3
SPLITS = ("train", "val")
OUTPUT_DIR_NAME = "owned_vocoder_loss_edge_semantics_forensics_v1"


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


def _centered_stft_magnitude(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    window = torch.hann_window(win_length, dtype=waveform.dtype, device=waveform.device)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    ).abs()


def _artificial_context_mask(
    *,
    sample_count: int,
    frame_count: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Frames whose centered FFT window reaches beyond the crop sample interval."""

    half = n_fft // 2
    centers = torch.arange(frame_count, dtype=torch.long) * int(hop_length)
    return (centers < half) | (centers + half > int(sample_count))


def _frame_error_summary(
    local: torch.Tensor,
    full_slice: torch.Tensor,
    artificial_mask: torch.Tensor,
) -> dict[str, float | int]:
    if local.shape != full_slice.shape:
        raise ValueError("local/full analysis shapes differ")
    if local.ndim != 2 or int(local.shape[1]) != int(artificial_mask.numel()):
        raise ValueError("analysis must be [frequency, frames] with matching mask")
    error = (
        torch.log(local.clamp_min(1e-5))
        - torch.log(full_slice.clamp_min(1e-5))
    ).abs().mean(dim=0)
    interior = ~artificial_mask
    return {
        "frames": int(error.numel()),
        "artificial_context_frames": int(artificial_mask.sum()),
        "artificial_context_fraction": float(artificial_mask.to(torch.float32).mean()),
        "all_log_magnitude_l1": float(error.mean()),
        "artificial_log_magnitude_l1": (
            float(error[artificial_mask].mean()) if bool(artificial_mask.any()) else 0.0
        ),
        "interior_log_magnitude_l1": (
            float(error[interior].mean()) if bool(interior.any()) else 0.0
        ),
    }


def _mel_crop_vs_cached_summary(
    crop_log_mel: torch.Tensor,
    cached_segment_mel: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    sample_count: int,
) -> dict[str, float | int | bool]:
    """Compare crop-local centered mel frames against the cached full-utterance slice."""

    if crop_log_mel.ndim != 2 or cached_segment_mel.ndim != 2:
        raise ValueError("mel tensors must be [frames, mel_bins]")
    conditioning_frames = int(cached_segment_mel.shape[0])
    if int(crop_log_mel.shape[0]) < conditioning_frames:
        raise ValueError("crop mel has fewer frames than conditioning")
    aligned = crop_log_mel[:conditioning_frames]
    if aligned.shape != cached_segment_mel.shape:
        raise ValueError("aligned crop/cached mel shapes differ")
    frame_error = (aligned - cached_segment_mel).abs().mean(dim=1)
    mask = _artificial_context_mask(
        sample_count=sample_count,
        frame_count=conditioning_frames,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    interior = ~mask
    return {
        "conditioning_frames": conditioning_frames,
        "crop_local_frames": int(crop_log_mel.shape[0]),
        "extra_crop_local_frames_without_conditioning": int(crop_log_mel.shape[0]) - conditioning_frames,
        "artificial_context_conditioning_frames": int(mask.sum()),
        "all_log_mel_l1": float(frame_error.mean()),
        "artificial_log_mel_l1": float(frame_error[mask].mean()) if bool(mask.any()) else 0.0,
        "interior_log_mel_l1": float(frame_error[interior].mean()) if bool(interior.any()) else 0.0,
        "has_unconditioned_terminal_analysis_frame": int(crop_log_mel.shape[0]) > conditioning_frames,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def run_owned_vocoder_loss_edge_forensics(
    root: Path,
    *,
    segment_mel_frames: int = DEFAULT_SEGMENT_MEL_FRAMES,
    items_per_split: int = DEFAULT_ITEMS_PER_SPLIT,
    seed: int = 4242,
) -> dict[str, object]:
    root = Path(root).resolve()
    speech_config = LykenoxSpeechConfig()
    vocoder_config = LykenoxVocoderConfig(
        mel_bins=speech_config.mel_bins,
        sample_rate=speech_config.sample_rate,
        hop_length=speech_config.hop_length,
    )
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=speech_config.sample_rate,
        n_fft=speech_config.n_fft,
        hop_length=speech_config.hop_length,
        n_mels=speech_config.mel_bins,
        power=1.0,
        center=True,
    )

    protected = _protected(root)
    before = {name: _sha256(path) for name, path in protected.items()}
    items: list[dict[str, object]] = []
    stft_rows: dict[str, list[dict[str, Any]]] = {
        f"{n_fft}_{hop}_{win}": [] for n_fft, hop, win in STFT_RESOLUTIONS
    }
    mel_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        dataset = _dataset(root, split, speech_config)
        full_items = {str(dataset[i]["utterance_id"]): dataset[i] for i in range(len(dataset))}
        segments, _skipped = collect_owned_vocoder_segments(
            root,
            split,
            segment_mel_frames=segment_mel_frames,
            max_items=items_per_split,
            seed=seed,
        )
        for segment in segments:
            full_item = full_items[segment.utterance_id]
            full_waveform = _mono_waveform(Path(str(full_item["wav_path"])), vocoder_config)
            full_mel = full_item["mel"].to(torch.float32).contiguous()
            start = int(segment.start_frame)
            end = start + int(segment.mel_frames)
            if not torch.equal(segment.mel, full_mel[start:end]):
                raise RuntimeError("owned segment mel is not the cached full-utterance slice")

            sample_count = int(segment.waveform.numel())
            resolution_payload: dict[str, object] = {}
            for n_fft, analysis_hop, win_length in STFT_RESOLUTIONS:
                if (start * speech_config.hop_length) % analysis_hop != 0:
                    raise RuntimeError("segment start is not exact on loss STFT grid")
                local = _centered_stft_magnitude(
                    segment.waveform,
                    n_fft=n_fft,
                    hop_length=analysis_hop,
                    win_length=win_length,
                )
                full = _centered_stft_magnitude(
                    full_waveform,
                    n_fft=n_fft,
                    hop_length=analysis_hop,
                    win_length=win_length,
                )
                start_analysis_frame = (
                    start * speech_config.hop_length // analysis_hop
                )
                full_slice = full[
                    :,
                    start_analysis_frame : start_analysis_frame + int(local.shape[1]),
                ]
                if full_slice.shape != local.shape:
                    raise RuntimeError("full utterance lacks required loss-analysis context")
                mask = _artificial_context_mask(
                    sample_count=sample_count,
                    frame_count=int(local.shape[1]),
                    n_fft=n_fft,
                    hop_length=analysis_hop,
                )
                summary = _frame_error_summary(local, full_slice, mask)
                key = f"{n_fft}_{analysis_hop}_{win_length}"
                stft_rows[key].append(summary)
                resolution_payload[key] = {
                    k: round(float(v), 8) if isinstance(v, float) else v
                    for k, v in summary.items()
                }

            crop_log_mel = torch.log(
                mel_transform(segment.waveform.unsqueeze(0)).clamp_min(1e-5)
            ).squeeze(0).transpose(0, 1)
            mel_summary = _mel_crop_vs_cached_summary(
                crop_log_mel,
                segment.mel,
                n_fft=speech_config.n_fft,
                hop_length=speech_config.hop_length,
                sample_count=sample_count,
            )
            mel_rows.append(mel_summary)
            items.append(
                {
                    "split": split,
                    "utterance_id": segment.utterance_id,
                    "start_frame": start,
                    "segment_frames": int(segment.mel_frames),
                    "stft": resolution_payload,
                    "mel": {
                        k: round(float(v), 8) if isinstance(v, float) else v
                        for k, v in mel_summary.items()
                    },
                }
            )

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after

    stft_mean: dict[str, object] = {}
    artificial_stft_observed = False
    interior_stft_exact = True
    for key, rows in stft_rows.items():
        mean_artificial = _mean(rows, "artificial_log_magnitude_l1")
        mean_interior = _mean(rows, "interior_log_magnitude_l1")
        artificial_stft_observed = artificial_stft_observed or mean_artificial > 1e-6
        interior_stft_exact = interior_stft_exact and mean_interior < 1e-6
        stft_mean[key] = {
            "mean_artificial_context_fraction": round(
                _mean(rows, "artificial_context_fraction"), 8
            ),
            "mean_all_log_magnitude_l1": round(
                _mean(rows, "all_log_magnitude_l1"), 8
            ),
            "mean_artificial_log_magnitude_l1": round(mean_artificial, 8),
            "mean_interior_log_magnitude_l1": round(mean_interior, 8),
        }

    mel_artificial = _mean(mel_rows, "artificial_log_mel_l1")
    mel_interior = _mean(mel_rows, "interior_log_mel_l1")
    extra_terminal = all(
        int(row["extra_crop_local_frames_without_conditioning"]) == 1
        for row in mel_rows
    )
    centered_crop_edge_bias_observed = (
        artificial_stft_observed
        and interior_stft_exact
        and mel_artificial > 1e-6
        and mel_interior < 1e-6
        and extra_terminal
    )
    status_pass = checkpoints_unchanged and centered_crop_edge_bias_observed

    output_dir = (
        root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    )
    report_path = output_dir / "loss_edge_semantics_forensics.json"
    report: dict[str, object] = {
        "status": "pass" if status_pass else "fail",
        "audit_version": AUDIT_VERSION,
        "owned_segment_contract": OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
        "segment_mel_frames": segment_mel_frames,
        "items_per_split": items_per_split,
        "centered_crop_edge_bias_observed": centered_crop_edge_bias_observed,
        "stft_interior_matches_full_utterance": interior_stft_exact,
        "stft_artificial_boundary_mismatch_observed": artificial_stft_observed,
        "mean_stft_by_resolution": stft_mean,
        "mel_crop_local_frame_count": segment_mel_frames + 1,
        "mel_conditioning_frame_count": segment_mel_frames,
        "mel_extra_terminal_frame_without_conditioning": extra_terminal,
        "mean_mel_all_log_l1": round(_mean(mel_rows, "all_log_mel_l1"), 8),
        "mean_mel_artificial_log_l1": round(mel_artificial, 8),
        "mean_mel_interior_log_l1": round(mel_interior, 8),
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
            "define_owned_vocoder_loss_v2_valid_interior_and_conditioning_aligned"
            if status_pass
            else "inspect_owned_loss_analysis_geometry_before_model_work"
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
            run_owned_vocoder_loss_edge_forensics(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
