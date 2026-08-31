"""Non-persistent held-out product smoke for the frame-hidden mel detail head.

The detail head is optimized in memory on a tiny fixed train subset, then evaluated on
three held-out utterances through accepted vocoder v4.2 with target prosody. The gate is
intentionally stronger than the rejected postnet smoke: improvement must generalize to
held-out mel fidelity and waveform presence, with no large per-item presence regression or
frame-grid artifact. No persistent checkpoint is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech.mel_detail_head import (
    MEL_DETAIL_HEAD_ARCHITECTURE_V1,
    LykenoxAcousticFrameHiddenDetailCandidate,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_audit import (
    _mel_bin_centers_hz,
    _mel_metrics,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_loss import (
    acoustic_mel_fidelity_loss,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import frame_grid_artifact_metrics
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import target_relative_presence_loss
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _load_reference_waveform,
)


SMOKE_VERSION = "acoustic-frame-hidden-mel-detail-heldout-product-smoke-v1"
TRAIN_INDICES = (0, 1, 2, 3, 4, 5, 6, 7)
VALIDATION_INDICES = (0, 1, 2)
UPDATES = 32
LEARNING_RATE = 3e-4
MAX_PRESENCE_REGRESSION_DB = 0.25


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
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "rejected_mel_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "rejected_postnet_best": training / "acoustic_mel_postnet_v1" / "best.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def _loss(candidate, batch):
    output = candidate(batch.token_ids, batch.token_mask, batch.durations)
    if not torch.equal(output["regulated_durations"], batch.durations):
        raise RuntimeError("detail-head smoke changed teacher durations")
    return output, acoustic_mel_fidelity_loss(
        output["mel"],
        batch.mel,
        batch.mel_mask,
        sample_rate=candidate.config.sample_rate,
        n_fft=candidate.config.n_fft,
    )


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average empty metric list")
    return sum(values) / len(values)


def run_mel_detail_head_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    for required in ("acoustic_v2_best", "v4_2_best"):
        if not protected[required].exists():
            raise FileNotFoundError(f"required checkpoint missing: {protected[required]}")
    before = {name: _sha256(path) for name, path in protected.items()}

    base, _base_payload = load_acoustic_prosody_checkpoint(protected["acoustic_v2_best"])
    vocoder, _disc, _vocoder_payload = load_v4_2_checkpoint(protected["v4_2_best"])
    base.cpu().eval()
    vocoder.cpu().eval()
    candidate = LykenoxAcousticFrameHiddenDetailCandidate(base).cpu().eval()
    trainable = candidate.trainable_parameter_names()
    only_detail_head_trainable = bool(trainable) and all(
        name.startswith("detail_head.") for name in trainable
    )
    if not only_detail_head_trainable:
        raise RuntimeError("detail-head smoke may optimize only detail_head parameters")
    if any(parameter.requires_grad for parameter in candidate.base_model.parameters()):
        raise RuntimeError("accepted acoustic base became trainable")

    duration_root = find_clean_duration_root(root)
    train_dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        base.config,
        duration_root=duration_root,
        include_pitch_targets=False,
    )
    val_dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        base.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(train_dataset) <= max(TRAIN_INDICES) or len(val_dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("detail-head smoke dataset is smaller than fixed smoke indices")

    centers = _mel_bin_centers_hz(
        sample_rate=base.config.sample_rate,
        n_fft=base.config.n_fft,
        mel_bins=base.config.mel_bins,
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    def evaluate() -> tuple[dict[str, float], list[dict[str, float | bool]]]:
        totals: list[float] = []
        mel_l1s: list[float] = []
        spectral_ratios: list[float] = []
        temporal_ratios: list[float] = []
        high_band_db: list[float] = []
        presence_errors: list[float] = []
        item_reports: list[dict[str, float | bool]] = []
        with torch.no_grad():
            for dataset_index in VALIDATION_INDICES:
                item = val_dataset[dataset_index]
                batch = collate_aligned_speech([item]).to("cpu")
                if batch.f0_hz is None or batch.voiced is None:
                    raise RuntimeError("detail-head heldout gate requires target F0/voicing")
                base_output = base(batch.token_ids, batch.token_mask, batch.durations)
                output, loss_result = _loss(candidate, batch)
                exact_non_mel = all(
                    torch.equal(base_output[key], output[key])
                    for key in (
                        "duration_prediction",
                        "regulated_durations",
                        "f0_prediction_hz",
                        "voicing_logits",
                        "mel_lengths",
                        "mel_mask",
                    )
                )
                if not exact_non_mel or not torch.equal(base_output["mel"], output["base_mel"]):
                    raise RuntimeError("detail head changed base/non-mel acoustic outputs")
                frames = int(batch.mel_lengths[0])
                metrics = _mel_metrics(
                    output["mel"][0, :frames],
                    batch.mel[0, :frames],
                    centers_hz=centers,
                )
                samples = frames * base.config.hop_length
                reference = _load_reference_waveform(
                    Path(str(val_dataset.base[dataset_index]["wav_path"])),
                    sample_rate=base.config.sample_rate,
                    samples=samples,
                )
                wave = vocoder(
                    output["mel"][:, :frames],
                    batch.f0_hz[:, :frames],
                    batch.voiced[:, :frames],
                )[0]
                presence = target_relative_presence_loss(
                    wave.unsqueeze(0),
                    reference.unsqueeze(0),
                    sample_rate=base.config.sample_rate,
                    hop_length=base.config.hop_length,
                )
                grid = frame_grid_artifact_metrics(
                    wave,
                    sample_rate=base.config.sample_rate,
                    hop_length=base.config.hop_length,
                )
                total = float(loss_result.total.detach())
                mel_l1 = float(loss_result.mel_l1.detach())
                presence_error = abs(float(presence.presence_1k_8k_error_db.detach()))
                totals.append(total)
                mel_l1s.append(mel_l1)
                spectral_ratios.append(float(metrics["spectral_delta_ratio"]))
                temporal_ratios.append(float(metrics["temporal_delta_ratio"]))
                high_band_db.append(float(metrics["band_3k_8k_relative_db"]))
                presence_errors.append(presence_error)
                item_reports.append({
                    "dataset_index": float(dataset_index),
                    "total": total,
                    "mel_l1": mel_l1,
                    "spectral_delta_ratio": float(metrics["spectral_delta_ratio"]),
                    "temporal_delta_ratio": float(metrics["temporal_delta_ratio"]),
                    "band_3k_8k_relative_db": float(metrics["band_3k_8k_relative_db"]),
                    "presence_1k_8k_error_db": presence_error,
                    "grid_failure": bool(grid.severe_grid_artifact[0]),
                })
        return {
            "total": _mean(totals),
            "mel_l1": _mean(mel_l1s),
            "spectral_delta_ratio": _mean(spectral_ratios),
            "temporal_delta_ratio": _mean(temporal_ratios),
            "band_3k_8k_relative_db": _mean(high_band_db),
            "presence_1k_8k_error_db": _mean(presence_errors),
        }, item_reports

    initial, initial_items = evaluate()
    zero_init_mel_exact = True
    for dataset_index in VALIDATION_INDICES:
        batch = collate_aligned_speech([val_dataset[dataset_index]]).to("cpu")
        with torch.no_grad():
            base_output = base(batch.token_ids, batch.token_mask, batch.durations)
            candidate_output = candidate(batch.token_ids, batch.token_mask, batch.durations)
        zero_init_mel_exact = zero_init_mel_exact and torch.equal(
            base_output["mel"], candidate_output["mel"]
        )
    if not zero_init_mel_exact:
        raise RuntimeError("zero-init detail head did not preserve accepted mel")

    optimizer = torch.optim.AdamW(candidate.detail_head.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    finite_gradients = True
    for update in range(UPDATES):
        dataset_index = TRAIN_INDICES[update % len(TRAIN_INDICES)]
        batch = collate_aligned_speech([train_dataset[dataset_index]]).to("cpu")
        optimizer.zero_grad(set_to_none=True)
        _output, result = _loss(candidate, batch)
        if not torch.isfinite(result.total):
            raise RuntimeError("non-finite detail-head smoke loss")
        result.total.backward()
        for parameter in candidate.detail_head.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                finite_gradients = False
        if not finite_gradients:
            raise RuntimeError("non-finite detail-head smoke gradient")
        torch.nn.utils.clip_grad_norm_(candidate.detail_head.parameters(), 5.0)
        optimizer.step()

    final, final_items = evaluate()
    mean_total_decreased = final["total"] < initial["total"]
    mean_mel_l1_decreased = final["mel_l1"] < initial["mel_l1"]
    spectral_ratio_closer = abs(final["spectral_delta_ratio"] - 1.0) < abs(
        initial["spectral_delta_ratio"] - 1.0
    )
    temporal_ratio_closer = abs(final["temporal_delta_ratio"] - 1.0) < abs(
        initial["temporal_delta_ratio"] - 1.0
    )
    high_band_closer = abs(final["band_3k_8k_relative_db"]) < abs(
        initial["band_3k_8k_relative_db"]
    )
    mean_presence_closer = final["presence_1k_8k_error_db"] < initial["presence_1k_8k_error_db"]
    presence_improved_items = 0
    no_large_presence_regression = True
    no_grid_failure = True
    for before_item, after_item in zip(initial_items, final_items, strict=True):
        before_error = float(before_item["presence_1k_8k_error_db"])
        after_error = float(after_item["presence_1k_8k_error_db"])
        if after_error < before_error:
            presence_improved_items += 1
        if after_error - before_error > MAX_PRESENCE_REGRESSION_DB:
            no_large_presence_regression = False
        if bool(after_item["grid_failure"]):
            no_grid_failure = False

    with torch.no_grad():
        check_batch = collate_aligned_speech([val_dataset[VALIDATION_INDICES[0]]]).to("cpu")
        base_output = base(check_batch.token_ids, check_batch.token_mask, check_batch.durations)
        final_output = candidate(check_batch.token_ids, check_batch.token_mask, check_batch.durations)
    duration_outputs_exact = (
        torch.equal(base_output["duration_prediction"], final_output["duration_prediction"])
        and torch.equal(base_output["regulated_durations"], final_output["regulated_durations"])
    )
    f0_voicing_outputs_exact = (
        torch.equal(base_output["f0_prediction_hz"], final_output["f0_prediction_hz"])
        and torch.equal(base_output["voicing_logits"], final_output["voicing_logits"])
    )

    after = {name: _sha256(path) for name, path in protected.items()}
    protected_unchanged = before == after
    status = "pass" if all((
        only_detail_head_trainable,
        zero_init_mel_exact,
        duration_outputs_exact,
        f0_voicing_outputs_exact,
        finite_gradients,
        mean_total_decreased,
        mean_mel_l1_decreased,
        spectral_ratio_closer,
        temporal_ratio_closer,
        high_band_closer,
        mean_presence_closer,
        presence_improved_items >= 2,
        no_large_presence_regression,
        no_grid_failure,
        protected_unchanged,
    )) else "fail"
    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "architecture": MEL_DETAIL_HEAD_ARCHITECTURE_V1,
        "train_indices": list(TRAIN_INDICES),
        "validation_indices": list(VALIDATION_INDICES),
        "updates": UPDATES,
        "only_detail_head_trainable": only_detail_head_trainable,
        "zero_init_mel_exact": zero_init_mel_exact,
        "duration_outputs_exact": duration_outputs_exact,
        "f0_voicing_outputs_exact": f0_voicing_outputs_exact,
        "finite_gradients": finite_gradients,
        "mean_total": f"{initial['total']:.6f} -> {final['total']:.6f}",
        "mean_mel_l1": f"{initial['mel_l1']:.6f} -> {final['mel_l1']:.6f}",
        "mean_spectral_delta_ratio": f"{initial['spectral_delta_ratio']:.6f} -> {final['spectral_delta_ratio']:.6f}",
        "mean_temporal_delta_ratio": f"{initial['temporal_delta_ratio']:.6f} -> {final['temporal_delta_ratio']:.6f}",
        "mean_band_3k_8k_relative_db": f"{initial['band_3k_8k_relative_db']:.6f} -> {final['band_3k_8k_relative_db']:.6f}",
        "mean_v4_2_presence_1k_8k_error_db": f"{initial['presence_1k_8k_error_db']:.6f} -> {final['presence_1k_8k_error_db']:.6f}",
        "presence_improved_items": presence_improved_items,
        "no_large_presence_regression": no_large_presence_regression,
        "no_grid_failure": no_grid_failure,
        "protected_checkpoints_unchanged": protected_unchanged,
        "persistent_training_started": False,
        "training_authorized": False,
        "next_gate": (
            "build_exact_resume_frame_hidden_detail_candidate"
            if status == "pass"
            else "reject_or_revise_frame_hidden_detail_before_persistent_training"
        ),
        "heldout_items_before": initial_items,
        "heldout_items_after": final_items,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_mel_detail_head_smoke(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
