"""CPU-first resumable trainer for the LYKENOX continuous residual source.

This is the replacement training path after closing discrete CELP/codebook retrieval.  Targets are
owned Step-3f real residuals extracted from TRAIN only.  The model predicts continuous 512-sample
sqrt-Hann residual vectors autoregressively at the 256-sample frame hop.  Validation is free-running.

The fixed minimum-phase renderer is used unchanged.  During source training its oracle cepstrum is a
TRAIN/VAL diagnostic target produced by the same Step-3f extraction; it is never a product inference
input and never enters the source model.  No third-party model/weight/service, post-hoc normalization,
EQ, denoise, duration modification, or codebook retrieval is used. Policy: LYX-POL-001.
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
import torch.nn.functional as F

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v1 import (
    CONTINUOUS_SOURCE_ARCHITECTURE,
    HOP_LENGTH,
    RESIDUAL_VECTOR_SAMPLES,
    LykenoxContinuousResidualSourceV1,
)
from lykenox_voice_engine.training.speech_glottal_calibration import extract_owned_real_residual
from lykenox_voice_engine.training.speech_residual_codebook_v1 import _sqrt_hann, residual_analysis_vectors
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    CEPSTRAL_ORDER,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)


TRAINER_VERSION = "owned-continuous-residual-source-trainer-v1"
TARGET_CACHE_VERSION = "owned-step3f-residual-target-cache-v1"
CHECKPOINT_SCHEMA_VERSION = "owned-continuous-residual-source-checkpoint-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 3
DEFAULT_SEGMENT_FRAMES = 64
DEFAULT_MAX_UPDATES = 600
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 25
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_SEED = 20260907
EPSILON = 1.0e-7


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_cache_path(root: Path, split: str, utterance_id: str) -> Path:
    return (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "continuous_residual_source_v1"
        / "targets"
        / split
        / f"{utterance_id}.pt"
    )


def _load_or_build_target(root: Path, utterance: OwnedVocoderUtterance) -> dict[str, Any]:
    path = _target_cache_path(root, utterance.split, utterance.utterance_id)
    if path.exists():
        try:
            cached = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            cached = torch.load(path, map_location="cpu")
        if (
            isinstance(cached, dict)
            and cached.get("target_cache_version") == TARGET_CACHE_VERSION
            and cached.get("utterance_id") == utterance.utterance_id
            and cached.get("split") == utterance.split
            and cached.get("wav_sha256") == _sha256(Path(utterance.wav_path))
        ):
            return cached

    residual, cepstrum, extension_frames = extract_owned_real_residual(
        utterance.waveform.cpu(), frame_count=int(utterance.mel_frames)
    )
    vectors = residual_analysis_vectors(residual)
    expected_vectors = int(utterance.mel_frames) + 1
    if vectors.shape != (expected_vectors, RESIDUAL_VECTOR_SAMPLES):
        raise RuntimeError("continuous-source target vector geometry changed")
    if cepstrum.shape != (int(utterance.mel_frames), CEPSTRAL_ORDER):
        raise RuntimeError("continuous-source target cepstrum geometry changed")
    payload: dict[str, Any] = {
        "target_cache_version": TARGET_CACHE_VERSION,
        "policy_id": POLICY_ID,
        "split": utterance.split,
        "utterance_id": utterance.utterance_id,
        "wav_path": utterance.wav_path,
        "wav_sha256": _sha256(Path(utterance.wav_path)),
        "frame_count": int(utterance.mel_frames),
        "extension_frames": int(extension_frames),
        "renderer_version": RENDERER_VERSION,
        "residual_vectors": vectors.to(torch.float32).contiguous(),
        "residual": residual.to(torch.float32).contiguous(),
        "cepstrum": cepstrum.to(torch.float32).contiguous(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return payload


def _ola_vectors(vectors: torch.Tensor, *, output_samples: int) -> torch.Tensor:
    """Differentiable counterpart of residual_synthesis_from_analysis_vectors."""
    if vectors.ndim != 3 or vectors.shape[-1] != RESIDUAL_VECTOR_SAMPLES:
        raise ValueError("vectors must be [batch, frames+1, 512]")
    expected = output_samples // HOP_LENGTH + 1
    if vectors.shape[1] != expected or output_samples % HOP_LENGTH:
        raise ValueError("vector/output geometry mismatch")
    window = _sqrt_hann(dtype=vectors.dtype).to(vectors.device)
    weighted = vectors * window.view(1, 1, -1)
    padded_samples = output_samples + 2 * HOP_LENGTH
    folded = F.fold(
        weighted.transpose(1, 2),
        output_size=(1, padded_samples),
        kernel_size=(1, RESIDUAL_VECTOR_SAMPLES),
        stride=(1, HOP_LENGTH),
    )
    padded = folded[:, 0, 0, :]
    return padded[:, HOP_LENGTH : HOP_LENGTH + output_samples].contiguous()


def _vector_shape_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = torch.sqrt(prediction.square().sum(dim=-1).clamp_min(1.0e-10))
    target_norm = torch.sqrt(target.square().sum(dim=-1).clamp_min(1.0e-10))
    cosine = (prediction * target).sum(dim=-1) / (pred_norm * target_norm)
    return (1.0 - cosine.clamp(-1.0, 1.0)).mean()


def _vector_energy_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rms = torch.sqrt(prediction.square().mean(dim=-1).clamp_min(1.0e-10))
    target_rms = torch.sqrt(target.square().mean(dim=-1).clamp_min(1.0e-10))
    return F.smooth_l1_loss(torch.log(pred_rms + EPSILON), torch.log(target_rms + EPSILON))


def _relative_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
    return ((prediction - target).abs() / scale).mean()


def _log_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(n_fft, dtype=prediction.dtype, device=prediction.device)
        pred = torch.stft(
            prediction,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs()
        ref = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs()
        losses.append(F.l1_loss(torch.log1p(pred), torch.log1p(ref)))
    return torch.stack(losses).mean()


def _teacher_ratio(update: int, max_updates: int) -> float:
    fraction = float(update) / float(max(max_updates, 1))
    if fraction <= 0.25:
        return 1.0
    if fraction <= 0.60:
        return 0.5
    return 0.0


def _deterministic_crop(frame_count: int, segment_frames: int, *, update: int, seed: int) -> int:
    if frame_count <= segment_frames:
        return 0
    span = frame_count - segment_frames
    value = (int(seed) * 1000003 + int(update) * 9176 + 37) % (span + 1)
    return int(value)


def _segment(
    utterance: OwnedVocoderUtterance,
    target: dict[str, Any],
    *,
    start: int,
    frames: int,
) -> tuple[torch.Tensor, ...]:
    end = start + frames
    waveform_start = start * HOP_LENGTH
    waveform_end = end * HOP_LENGTH
    return (
        utterance.mel[start:end].unsqueeze(0).cpu(),
        utterance.f0_hz[start:end].unsqueeze(0).cpu(),
        utterance.voiced[start:end].unsqueeze(0).cpu(),
        utterance.periodicity[start:end].unsqueeze(0).cpu(),
        target["residual_vectors"][start : end + 1].unsqueeze(0).cpu(),
        target["residual"][waveform_start:waveform_end].unsqueeze(0).cpu(),
        target["cepstrum"][start:end].unsqueeze(0).cpu(),
        utterance.waveform[waveform_start:waveform_end].unsqueeze(0).cpu(),
    )


def _loss_terms(
    model: LykenoxContinuousResidualSourceV1,
    tensors: tuple[torch.Tensor, ...],
    *,
    teacher_ratio: float,
    teacher_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    mel, f0_hz, voiced, periodicity, target_vectors, target_residual, oracle_cepstrum, reference = tensors
    predicted_vectors = model(
        mel,
        f0_hz,
        voiced,
        periodicity,
        teacher_vectors=target_vectors,
        teacher_forcing_ratio=teacher_ratio,
        teacher_seed=teacher_seed,
    )
    output_samples = int(mel.shape[1]) * HOP_LENGTH
    predicted_residual = _ola_vectors(predicted_vectors, output_samples=output_samples)
    prediction = render_time_varying_minimum_phase(
        predicted_residual,
        oracle_cepstrum,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )
    if prediction.shape != reference.shape:
        raise RuntimeError("continuous source renderer output geometry changed")

    shape = _vector_shape_loss(predicted_vectors, target_vectors)
    energy = _vector_energy_loss(predicted_vectors, target_vectors)
    residual_l1 = _relative_l1(predicted_residual, target_residual)
    waveform_l1 = _relative_l1(prediction, reference)
    spectral = _log_stft_loss(prediction, reference)
    total = shape + 0.25 * energy + 0.75 * residual_l1 + waveform_l1 + 0.5 * spectral
    public = {
        "total": float(total.detach()),
        "vector_shape": float(shape.detach()),
        "vector_log_rms": float(energy.detach()),
        "residual_relative_l1": float(residual_l1.detach()),
        "waveform_relative_l1": float(waveform_l1.detach()),
        "waveform_log_stft": float(spectral.detach()),
    }
    return total, public


@torch.no_grad()
def _evaluate(
    model: LykenoxContinuousResidualSourceV1,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
    *,
    segment_frames: int,
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for index, utterance in enumerate(utterances):
        target = _load_or_build_target(root, utterance)
        frames = min(segment_frames, int(utterance.mel_frames))
        start = max(0, (int(utterance.mel_frames) - frames) // 2)
        tensors = _segment(utterance, target, start=start, frames=frames)
        _, terms = _loss_terms(
            model,
            tensors,
            teacher_ratio=0.0,
            teacher_seed=100000 + index,
        )
        totals.append(terms)
    keys = tuple(totals[0])
    return {key: sum(item[key] for item in totals) / float(len(totals)) for key in keys}


def _save_checkpoint(
    path: Path,
    model: LykenoxContinuousResidualSourceV1,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    best_val: float,
    config: dict[str, object],
) -> None:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "update": int(update),
        "best_val_total": float(best_val),
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_continuous_residual_source(
    root: Path,
    *,
    train_items: int = DEFAULT_TRAIN_ITEMS,
    val_items: int = DEFAULT_VAL_ITEMS,
    segment_frames: int = DEFAULT_SEGMENT_FRAMES,
    max_updates: int = DEFAULT_MAX_UPDATES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    resume: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if segment_frames < 16 or max_updates < 1 or train_items < 1 or val_items < 1:
        raise ValueError("invalid continuous-source training limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v1"
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    config: dict[str, object] = {
        "train_items": train_items,
        "val_items": val_items,
        "segment_frames": segment_frames,
        "max_updates": max_updates,
        "learning_rate": learning_rate,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
    }

    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
    model = LykenoxContinuousResidualSourceV1().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    start_update = 0
    best_val = math.inf
    if resume and latest.exists():
        try:
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest, map_location="cpu")
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("continuous-source checkpoint schema mismatch")
        if checkpoint.get("architecture") != CONTINUOUS_SOURCE_ARCHITECTURE:
            raise RuntimeError("continuous-source checkpoint architecture mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_update = int(checkpoint["update"])
        best_val = float(checkpoint.get("best_val_total", math.inf))

    history_path = run_dir / "history.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    for update in range(start_update + 1, max_updates + 1):
        model.train()
        utterance_index = (int(seed) + update * 7919) % len(train_set)
        utterance = train_set[utterance_index]
        target = _load_or_build_target(root, utterance)
        frames = min(segment_frames, int(utterance.mel_frames))
        start = _deterministic_crop(
            int(utterance.mel_frames), frames, update=update, seed=seed + utterance_index
        )
        tensors = _segment(utterance, target, start=start, frames=frames)
        ratio = _teacher_ratio(update, max_updates)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss_terms(
            model,
            tensors,
            teacher_ratio=ratio,
            teacher_seed=seed + update,
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("continuous-source gradient norm became non-finite")
        optimizer.step()

        record: dict[str, object] = {
            "update": update,
            "utterance_id": utterance.utterance_id,
            "start_frame": start,
            "teacher_forcing_ratio": ratio,
            "grad_norm": grad_norm,
            **terms,
        }
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = _evaluate(model, root, val_set, segment_frames=segment_frames)
            record["validation"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(
                    best, model, optimizer, update=update, best_val=best_val, config=config
                )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(
                latest, model, optimizer, update=update, best_val=best_val, config=config
            )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "continuous_residual_source_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "training_split_only_for_optimizer_updates": True,
        "codebook_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_complete_heldout_utterances_with_best_checkpoint_and_listen",
    }
    _atomic_json(run_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-items", type=int, default=DEFAULT_TRAIN_ITEMS)
    parser.add_argument("--val-items", type=int, default=DEFAULT_VAL_ITEMS)
    parser.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--max-updates", type=int, default=DEFAULT_MAX_UPDATES)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = train_continuous_residual_source(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
