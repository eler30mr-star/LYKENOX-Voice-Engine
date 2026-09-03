"""CPU-first trainer for the LYKENOX non-overlapping continuous residual stream source V1.

Hop-grid forensics proved that learned 512/256 overlapping vectors violate duplicated-sample
consistency and concentrate high-band power on the 93.75 Hz hop grid before the fixed renderer.
This trainer removes that representation defect at the target/output contract: each acoustic frame
owns exactly one contiguous 256-sample Step-3f residual block and no sample is predicted twice.

To isolate the representation correction, the loss weights match Continuous Source V2 as closely as
possible: shape, explicit block level, residual relative-L1/level, rendered waveform relative-L1/
level and true-log STFT. There is no teacher forcing because no previous waveform block is an input.
The context encoder may be warm-started from the owned conditioning-V2 controlled checkpoint; the
new recurrent stream state and 256-sample head are trained from scratch.

No codebook, overlap-add source synthesis, external model/weight/service, post-hoc gain, EQ, denoise
or duration modification is used. Policy: LYX-POL-001. CPU reference device.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_continuous_stream_source_v1 import (
    BLOCK_SAMPLES,
    CONTINUOUS_STREAM_SOURCE_ARCHITECTURE,
    HOP_LENGTH,
    LykenoxContinuousResidualStreamSourceV1,
    blocks_to_residual,
)
from lykenox_voice_engine.training.speech_pitch_cache import PITCH_CONFIG
from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    PITCH_CONDITIONING_V2,
    PitchConditioningV2,
    extract_pitch_conditioning_v2,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import (
    _load_or_build_target,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_v2_pitch_conditioning_v2 import (
    CHECKPOINT_SCHEMA_VERSION as CONDITIONING_V2_CHECKPOINT_SCHEMA,
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


TRAINER_VERSION = "owned-continuous-residual-stream-source-trainer-v1"
CHECKPOINT_SCHEMA_VERSION = "owned-continuous-residual-stream-source-checkpoint-v1"
POLICY_ID = "LYX-POL-001"
RUN_DIR_NAME = "continuous_residual_stream_source_v1"
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 3
DEFAULT_SEGMENT_FRAMES = 96
DEFAULT_MAX_UPDATES = 600
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 50
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_SEED = 20260929
EPSILON = 1.0e-7


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _conditioning(utterance: OwnedVocoderUtterance) -> PitchConditioningV2:
    return extract_pitch_conditioning_v2(
        utterance.waveform.cpu().to(torch.float32),
        frame_count=int(utterance.mel_frames),
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        frame_length=int(PITCH_CONFIG["frame_length"]),
        min_f0_hz=float(PITCH_CONFIG["min_f0_hz"]),
        max_f0_hz=float(PITCH_CONFIG["max_f0_hz"]),
        anchor_periodicity_threshold=float(PITCH_CONFIG["voiced_periodicity_threshold"]),
        anchor_rms_fraction=float(PITCH_CONFIG["voiced_rms_fraction"]),
    )


def _deterministic_crop(frame_count: int, segment_frames: int, *, update: int, seed: int) -> int:
    if frame_count <= segment_frames:
        return 0
    span = frame_count - segment_frames
    return int((int(seed) * 1000003 + int(update) * 9176 + 83) % (span + 1))


def _segment(
    utterance: OwnedVocoderUtterance,
    conditioning: PitchConditioningV2,
    target: dict[str, Any],
    *,
    start: int,
    frames: int,
) -> tuple[torch.Tensor, ...]:
    end = start + frames
    sample_start = start * HOP_LENGTH
    sample_end = end * HOP_LENGTH
    residual = target["residual"][sample_start:sample_end].to(torch.float32).contiguous()
    if residual.numel() != frames * BLOCK_SAMPLES:
        raise RuntimeError("stream target segment geometry changed")
    blocks = residual.view(frames, BLOCK_SAMPLES).contiguous()
    return (
        utterance.mel[start:end].unsqueeze(0).cpu(),
        conditioning.f0_track_hz[start:end].unsqueeze(0).cpu(),
        conditioning.energy_confidence[start:end].unsqueeze(0).cpu(),
        conditioning.periodic_strength[start:end].unsqueeze(0).cpu(),
        blocks.unsqueeze(0).cpu(),
        residual.unsqueeze(0).cpu(),
        target["cepstrum"][start:end].unsqueeze(0).cpu(),
        utterance.waveform[sample_start:sample_end].unsqueeze(0).cpu(),
    )


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))


def _shape_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = torch.sqrt(prediction.square().sum(dim=-1).clamp_min(1.0e-10))
    target_norm = torch.sqrt(target.square().sum(dim=-1).clamp_min(1.0e-10))
    cosine = (prediction * target).sum(dim=-1) / (pred_norm * target_norm)
    return (1.0 - cosine.clamp(-1.0, 1.0)).mean()


def _target_log_rms(target: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.sqrt(target.square().mean(dim=-1).clamp_min(EPSILON * EPSILON)))


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
        losses.append(
            F.l1_loss(torch.log(pred.clamp_min(1.0e-5)), torch.log(ref.clamp_min(1.0e-5)))
        )
    return torch.stack(losses).mean()


def _loss_terms(
    model: LykenoxContinuousResidualStreamSourceV1,
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    mel, f0, energy, periodic, target_blocks, target_residual, oracle_cepstrum, reference = tensors
    predicted_blocks, predicted_log_rms = model.forward_with_log_rms(mel, f0, energy, periodic)
    predicted_residual = blocks_to_residual(predicted_blocks)
    prediction = render_time_varying_minimum_phase(
        predicted_residual,
        oracle_cepstrum,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )
    if predicted_residual.shape != target_residual.shape or prediction.shape != reference.shape:
        raise RuntimeError("continuous stream output geometry changed")

    shape = _shape_loss(predicted_blocks, target_blocks)
    block_level = F.smooth_l1_loss(predicted_log_rms, _target_log_rms(target_blocks))
    residual_l1 = _relative_l1(predicted_residual, target_residual)
    residual_level = _sequence_log_rms_loss(predicted_residual, target_residual)
    waveform_l1 = _relative_l1(prediction, reference)
    waveform_level = _sequence_log_rms_loss(prediction, reference)
    spectral = _true_log_stft_loss(prediction, reference)

    # Match V2 loss authority while changing only the residual representation geometry.
    total = (
        0.75 * shape
        + 1.25 * block_level
        + 0.75 * residual_l1
        + 1.00 * residual_level
        + 0.75 * waveform_l1
        + 1.25 * waveform_level
        + 0.50 * spectral
    )
    public = {
        "total": float(total.detach()),
        "block_shape": float(shape.detach()),
        "block_log_rms": float(block_level.detach()),
        "residual_relative_l1": float(residual_l1.detach()),
        "residual_log_rms": float(residual_level.detach()),
        "waveform_relative_l1": float(waveform_l1.detach()),
        "waveform_log_rms": float(waveform_level.detach()),
        "waveform_true_log_stft": float(spectral.detach()),
        "residual_rms_ratio": float((_rms(predicted_residual) / _rms(target_residual)).mean().detach()),
        "waveform_rms_ratio": float((_rms(prediction) / _rms(reference)).mean().detach()),
    }
    return total, public


def _warm_start_context(root: Path, model: LykenoxContinuousResidualStreamSourceV1) -> dict[str, object]:
    checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "continuous_residual_source_v2_pitch_conditioning_v2"
        / "best.pt"
    )
    result: dict[str, object] = {"used": False, "checkpoint": str(checkpoint), "loaded_tensor_count": 0}
    if not checkpoint.exists():
        return result
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != CONDITIONING_V2_CHECKPOINT_SCHEMA:
        raise RuntimeError("conditioning-v2 warm-start checkpoint schema mismatch")
    if payload.get("conditioning_contract") != PITCH_CONDITIONING_V2:
        raise RuntimeError("conditioning-v2 warm-start contract mismatch")
    state = payload["model_state"]
    own = model.state_dict()
    copied = 0
    for key, value in state.items():
        if not (key.startswith("conditioning_projection.") or key.startswith("context_blocks.")):
            continue
        if key in own and own[key].shape == value.shape:
            own[key] = value.detach().clone()
            copied += 1
    if copied < 2:
        raise RuntimeError("stream source warm start copied no meaningful context tensors")
    model.load_state_dict(own)
    result.update({"used": True, "loaded_tensor_count": copied})
    return result


@torch.no_grad()
def _evaluate_complete(
    model: LykenoxContinuousResidualStreamSourceV1,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for utterance in utterances:
        target = _load_or_build_target(root, utterance)
        conditioning = _conditioning(utterance)
        tensors = _segment(
            utterance,
            conditioning,
            target,
            start=0,
            frames=int(utterance.mel_frames),
        )
        _, public = _loss_terms(model, tensors)
        totals.append(public)
    keys = tuple(totals[0])
    return {key: sum(row[key] for row in totals) / float(len(totals)) for key in keys}


def _save_checkpoint(
    path: Path,
    model: LykenoxContinuousResidualStreamSourceV1,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    best_val: float,
    config: dict[str, object],
    warm_start: dict[str, object],
) -> None:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_STREAM_SOURCE_ARCHITECTURE,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "residual_representation": "unique_contiguous_256_sample_blocks_no_overlap",
        "update": int(update),
        "best_val_total": float(best_val),
        "config": config,
        "warm_start": warm_start,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_continuous_stream_source_v1(
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
        raise ValueError("invalid continuous stream training limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / RUN_DIR_NAME
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
    conditioning_cache = {item.utterance_id: _conditioning(item) for item in train_set + val_set}

    model = LykenoxContinuousResidualStreamSourceV1().cpu()
    warm_start = _warm_start_context(root, model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=DEFAULT_WEIGHT_DECAY
    )
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
        "block_samples": BLOCK_SAMPLES,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "overlap_samples": 0,
        "duplicated_sample_authority": False,
        "previous_waveform_feedback": False,
        "teacher_forcing": False,
        "loss_weights_match_continuous_source_v2": True,
        "complete_heldout_validation": True,
    }

    start_update = 0
    best_val = math.inf
    if resume and latest.exists():
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(latest, map_location="cpu")
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("continuous stream checkpoint schema mismatch")
        if payload.get("architecture") != CONTINUOUS_STREAM_SOURCE_ARCHITECTURE:
            raise RuntimeError("continuous stream architecture mismatch")
        if payload.get("conditioning_contract") != PITCH_CONDITIONING_V2:
            raise RuntimeError("continuous stream conditioning mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_update = int(payload["update"])
        best_val = float(payload.get("best_val_total", math.inf))
        warm_start = dict(payload.get("warm_start", warm_start))

    history = run_dir / "history.jsonl"
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
        tensors = _segment(
            utterance,
            conditioning_cache[utterance.utterance_id],
            target,
            start=start,
            frames=frames,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss_terms(model, tensors)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("continuous stream gradient norm became non-finite")
        optimizer.step()

        record: dict[str, object] = {
            "update": update,
            "utterance_id": utterance.utterance_id,
            "start_frame": start,
            "teacher_forcing": False,
            "overlap_samples": 0,
            "grad_norm": grad_norm,
            **terms,
        }
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = _evaluate_complete(model, root, val_set)
            record["validation_complete_utterances"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(
                    best,
                    model,
                    optimizer,
                    update=update,
                    best_val=best_val,
                    config=config,
                    warm_start=warm_start,
                )
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(
                latest,
                model,
                optimizer,
                update=update,
                best_val=best_val,
                config=config,
                warm_start=warm_start,
            )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "continuous_residual_stream_source_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": CONTINUOUS_STREAM_SOURCE_ARCHITECTURE,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "warm_start": warm_start,
        "residual_representation": "unique_contiguous_256_sample_blocks_no_overlap",
        "overlap_samples": 0,
        "duplicated_sample_authority": False,
        "teacher_forcing_used": False,
        "previous_waveform_feedback_used": False,
        "codebook_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render complete heldout stream source and compare against V2 plus identity ceiling",
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
    print(json.dumps(train_continuous_stream_source_v1(
        args.root,
        train_items=args.train_items,
        val_items=args.val_items,
        segment_frames=args.segment_frames,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        resume=not args.no_resume,
        seed=args.seed,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
