"""Production-oriented CPU trainer for the level-factored LYKENOX continuous residual source V2.

V1 exhibited a held-out amplitude collapse (prediction/reference RMS about 0.10-0.15).  V2 fixes
that failure structurally: residual fine-structure and residual level are separate predicted
variables.  The trainer therefore supervises vector shape, vector log-RMS, reconstructed residual
level, and rendered waveform level explicitly.  Validation is fully autoregressive and uses complete
held-out utterances so long-run level drift cannot be hidden by short crops.

Targets are owned Step-3f real residuals.  The minimum-phase renderer is unchanged.  No codebook,
post-hoc gain, EQ, denoise, external model/weight/service, or duration modification is used.
Policy: LYX-POL-001. CPU is the reference device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_source_v2 import (
    CONTINUOUS_SOURCE_ARCHITECTURE_V2,
    HOP_LENGTH,
    RESIDUAL_VECTOR_SAMPLES,
    LykenoxContinuousResidualSourceV2,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import (
    _load_or_build_target,
    _ola_vectors,
    _segment,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_time_varying_minimum_phase,
)


TRAINER_VERSION = "owned-continuous-residual-source-trainer-v2-level-factored"
CHECKPOINT_SCHEMA_VERSION = "owned-continuous-residual-source-checkpoint-v2-level-factored"
POLICY_ID = "LYX-POL-001"
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 3
DEFAULT_SEGMENT_FRAMES = 96
DEFAULT_MAX_UPDATES = 600
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 50
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_SEED = 20260911
EPSILON = 1.0e-7


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))


def _vector_shape_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = torch.sqrt(prediction.square().sum(dim=-1).clamp_min(1.0e-10))
    target_norm = torch.sqrt(target.square().sum(dim=-1).clamp_min(1.0e-10))
    cosine = (prediction * target).sum(dim=-1) / (pred_norm * target_norm)
    return (1.0 - cosine.clamp(-1.0, 1.0)).mean()


def _target_vector_log_rms(target: torch.Tensor) -> torch.Tensor:
    target_rms = torch.sqrt(target.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))
    return torch.log(target_rms)


def _relative_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
    return ((prediction - target).abs() / scale).mean()


def _sequence_log_rms_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(torch.log(_rms(prediction)), torch.log(_rms(target)))


def _true_log_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        pred_log = torch.log(pred.clamp_min(1.0e-5))
        ref_log = torch.log(ref.clamp_min(1.0e-5))
        losses.append(F.l1_loss(pred_log, ref_log))
    return torch.stack(losses).mean()


def _teacher_ratio(update: int, max_updates: int) -> float:
    fraction = float(update) / float(max(max_updates, 1))
    if fraction <= 0.10:
        return 1.0
    if fraction <= 0.25:
        return 0.50
    if fraction <= 0.40:
        return 0.25
    return 0.0


def _deterministic_crop(frame_count: int, segment_frames: int, *, update: int, seed: int) -> int:
    if frame_count <= segment_frames:
        return 0
    span = frame_count - segment_frames
    value = (int(seed) * 1000003 + int(update) * 9176 + 37) % (span + 1)
    return int(value)


def _loss_terms(
    model: LykenoxContinuousResidualSourceV2,
    tensors: tuple[torch.Tensor, ...],
    *,
    teacher_ratio: float,
    teacher_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    mel, f0_hz, voiced, periodicity, target_vectors, target_residual, oracle_cepstrum, reference = tensors
    predicted_vectors, predicted_log_rms = model.forward_with_log_rms(
        mel,
        f0_hz,
        voiced,
        periodicity,
        teacher_vectors=target_vectors,
        teacher_forcing_ratio=teacher_ratio,
        teacher_seed=teacher_seed,
    )
    target_log_rms = _target_vector_log_rms(target_vectors)
    output_samples = int(mel.shape[1]) * HOP_LENGTH
    predicted_residual = _ola_vectors(predicted_vectors, output_samples=output_samples)
    prediction = render_time_varying_minimum_phase(
        predicted_residual,
        oracle_cepstrum,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )
    if prediction.shape != reference.shape:
        raise RuntimeError("continuous source v2 renderer output geometry changed")

    shape = _vector_shape_loss(predicted_vectors, target_vectors)
    vector_level = F.smooth_l1_loss(predicted_log_rms, target_log_rms)
    residual_l1 = _relative_l1(predicted_residual, target_residual)
    residual_level = _sequence_log_rms_loss(predicted_residual, target_residual)
    waveform_l1 = _relative_l1(prediction, reference)
    waveform_level = _sequence_log_rms_loss(prediction, reference)
    spectral = _true_log_stft_loss(prediction, reference)

    # Level is first-class, not a weak auxiliary term. There is no output renormalization: the
    # model itself must generate the correct source energy before the frozen renderer.
    total = (
        0.75 * shape
        + 1.25 * vector_level
        + 0.75 * residual_l1
        + 1.00 * residual_level
        + 0.75 * waveform_l1
        + 1.25 * waveform_level
        + 0.50 * spectral
    )

    residual_ratio = float((_rms(predicted_residual) / _rms(target_residual)).mean().detach())
    waveform_ratio = float((_rms(prediction) / _rms(reference)).mean().detach())
    public = {
        "total": float(total.detach()),
        "vector_shape": float(shape.detach()),
        "vector_log_rms": float(vector_level.detach()),
        "residual_relative_l1": float(residual_l1.detach()),
        "residual_log_rms": float(residual_level.detach()),
        "waveform_relative_l1": float(waveform_l1.detach()),
        "waveform_log_rms": float(waveform_level.detach()),
        "waveform_true_log_stft": float(spectral.detach()),
        "residual_rms_ratio": residual_ratio,
        "waveform_rms_ratio": waveform_ratio,
    }
    return total, public


@torch.no_grad()
def _evaluate_complete(
    model: LykenoxContinuousResidualSourceV2,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for index, utterance in enumerate(utterances):
        target = _load_or_build_target(root, utterance)
        tensors = _segment(
            utterance,
            target,
            start=0,
            frames=int(utterance.mel_frames),
        )
        _, terms = _loss_terms(
            model,
            tensors,
            teacher_ratio=0.0,
            teacher_seed=200000 + index,
        )
        totals.append(terms)
    keys = tuple(totals[0])
    return {key: sum(item[key] for item in totals) / float(len(totals)) for key in keys}


def _save_checkpoint(
    path: Path,
    model: LykenoxContinuousResidualSourceV2,
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
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE_V2,
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


def train_continuous_residual_source_v2(
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
    if segment_frames < 32 or max_updates < 1 or train_items < 1 or val_items < 1:
        raise ValueError("invalid continuous-source v2 training limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2"
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
        "level_factorized": True,
        "previous_amplitude_recurrent_feedback": False,
        "complete_heldout_validation": True,
    }

    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
    model = LykenoxContinuousResidualSourceV2().cpu()
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
            raise RuntimeError("continuous-source v2 checkpoint schema mismatch")
        if checkpoint.get("architecture") != CONTINUOUS_SOURCE_ARCHITECTURE_V2:
            raise RuntimeError("continuous-source v2 checkpoint architecture mismatch")
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
            raise RuntimeError("continuous-source v2 gradient norm became non-finite")
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
            validation = _evaluate_complete(model, root, val_set)
            record["validation_complete_utterances"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(best, model, optimizer, update=update, best_val=best_val, config=config)

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(latest, model, optimizer, update=update, best_val=best_val, config=config)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "continuous_residual_source_v2_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_SOURCE_ARCHITECTURE_V2,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "v1_amplitude_collapse_root_fix": "explicit_shape_times_predicted_rms_factorization",
        "previous_amplitude_recurrent_feedback": False,
        "complete_heldout_validation": True,
        "training_split_only_for_optimizer_updates": True,
        "codebook_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_complete_heldout_utterances_from_level_factored_source",
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
    report = train_continuous_residual_source_v2(
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
