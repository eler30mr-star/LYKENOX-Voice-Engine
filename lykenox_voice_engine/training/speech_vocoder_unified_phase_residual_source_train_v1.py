"""CPU-first trainer for the LYKENOX unified phase-aware residual source V1.

The product-source problem is trained as one residual, not a handoff between two independently
learned sources.  One hidden state predicts a periodic Fourier coordinate and an aperiodic 512/256
OLA coordinate.  Fixed owned F0 phase evaluates the periodic coordinate; complementary energy
weights sqrt(voiced*periodicity) and sqrt(1-voiced*periodicity) make the two coordinates one source.

Stable voiced samples directly supervise the periodic coordinate, stable non-periodic samples
directly supervise the aperiodic coordinate, and transition samples are optimized only through the
joint reconstructed Step-3f residual and frozen minimum-phase renderer.  There is no teacher forcing,
codebook, stochastic innovation, source fallback/handoff, external model/weight/service, gain
normalization, EQ, denoise or duration modification. Policy: LYX-POL-001. CPU reference device.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_unified_phase_residual_source_v1 import (
    HARMONIC_COUNT,
    HOP_LENGTH,
    RESIDUAL_VECTOR_SAMPLES,
    UNIFIED_PHASE_SOURCE_ARCHITECTURE,
    LykenoxUnifiedPhaseResidualSourceV1,
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
    fixed_linear_frame_to_sample,
    render_time_varying_minimum_phase,
)
from lykenox_voice_engine.training.speech_vocoder_pitch_synchronous_cycle_source_train_v1 import (
    CHECKPOINT_SCHEMA_VERSION as PITCH_SYNC_CHECKPOINT_SCHEMA_VERSION,
)


TRAINER_VERSION = "owned-unified-phase-residual-source-trainer-v1"
CHECKPOINT_SCHEMA_VERSION = "owned-unified-phase-residual-source-checkpoint-v1"
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
DEFAULT_SEED = 20260923
EPSILON = 1.0e-7


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _deterministic_crop(frame_count: int, segment_frames: int, *, update: int, seed: int) -> int:
    if frame_count <= segment_frames:
        return 0
    span = frame_count - segment_frames
    return int((int(seed) * 1000003 + int(update) * 9176 + 71) % (span + 1))


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(value.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))


def _frame_to_sample_multi(value: torch.Tensor, *, hop_length: int = HOP_LENGTH) -> torch.Tensor:
    """Interpolate [B,F,...] frame values to [B,F*hop,...] with the fixed renderer rule."""
    if value.ndim < 2:
        raise ValueError("frame tensor must have at least batch and frame dimensions")
    batch, frames = int(value.shape[0]), int(value.shape[1])
    tail = tuple(value.shape[2:])
    flattened = value.permute(0, *range(2, value.ndim), 1).reshape(-1, frames)
    sampled = fixed_linear_frame_to_sample(flattened, hop_length=hop_length)
    sample_count = frames * hop_length
    restored = sampled.reshape(batch, *tail, sample_count)
    order = [0, len(tail) + 1, *range(1, len(tail) + 1)]
    return restored.permute(*order).contiguous()


def accumulated_phase_offset(utterance: OwnedVocoderUtterance, start_frame: int) -> float:
    """Exact fixed-interpolation F0 phase at a crop boundary, using conditioning only."""
    if start_frame <= 0:
        return 0.0
    sample_f0 = fixed_linear_frame_to_sample(
        utterance.f0_hz.unsqueeze(0).to(torch.float32), hop_length=HOP_LENGTH
    ).squeeze(0)
    sample_voiced = fixed_linear_frame_to_sample(
        utterance.voiced.unsqueeze(0).to(torch.float32), hop_length=HOP_LENGTH
    ).squeeze(0)
    stop = min(int(sample_f0.numel()), int(start_frame) * HOP_LENGTH)
    increment = torch.where(
        (sample_f0[:stop] > 0.0) & (sample_voiced[:stop] >= 0.5),
        sample_f0[:stop] / float(SAMPLE_RATE),
        torch.zeros_like(sample_f0[:stop]),
    )
    return float(increment.sum())


def synthesize_unified_residual(
    harmonic_pairs: torch.Tensor,
    aperiodic_vectors: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
    *,
    phase_offset_cycles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (joint_residual, periodic_wave, aperiodic_wave, periodic_strength_samples)."""
    if harmonic_pairs.ndim != 4 or harmonic_pairs.shape[-2:] != (HARMONIC_COUNT, 2):
        raise ValueError("harmonic_pairs must be [B,F,H,2]")
    batch, frames = harmonic_pairs.shape[:2]
    if f0_hz.shape != (batch, frames) or voiced.shape != (batch, frames) or periodicity.shape != (batch, frames):
        raise ValueError("conditioning geometry differs from harmonic frames")
    output_samples = frames * HOP_LENGTH
    if aperiodic_vectors.shape != (batch, frames + 1, RESIDUAL_VECTOR_SAMPLES):
        raise ValueError("aperiodic vector geometry changed")

    pair_samples = _frame_to_sample_multi(harmonic_pairs, hop_length=HOP_LENGTH)
    sample_f0 = fixed_linear_frame_to_sample(f0_hz.clamp_min(0.0), hop_length=HOP_LENGTH)
    sample_voiced = fixed_linear_frame_to_sample(voiced.clamp(0.0, 1.0), hop_length=HOP_LENGTH)
    sample_periodicity = fixed_linear_frame_to_sample(periodicity.clamp(0.0, 1.0), hop_length=HOP_LENGTH)
    periodic_strength = (sample_voiced * sample_periodicity).clamp(0.0, 1.0)

    phase_increment = torch.where(
        (sample_f0 > 0.0) & (sample_voiced >= 0.5),
        sample_f0 / float(SAMPLE_RATE),
        torch.zeros_like(sample_f0),
    )
    cumulative = torch.cumsum(phase_increment, dim=-1) - phase_increment
    if phase_offset_cycles is None:
        phase_offset_cycles = torch.zeros(batch, dtype=cumulative.dtype, device=cumulative.device)
    if phase_offset_cycles.shape != (batch,):
        raise ValueError("phase_offset_cycles must have shape [batch]")
    phase = cumulative + phase_offset_cycles.unsqueeze(-1)

    harmonic_index = torch.arange(
        1, HARMONIC_COUNT + 1, dtype=phase.dtype, device=phase.device
    ).view(1, 1, HARMONIC_COUNT)
    angle = 2.0 * math.pi * phase.unsqueeze(-1) * harmonic_index
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    periodic_wave = (
        pair_samples[..., 0] * cosine + pair_samples[..., 1] * sine
    ).sum(dim=-1)

    aperiodic_wave = _ola_vectors(aperiodic_vectors, output_samples=output_samples)
    periodic_weight = torch.sqrt(periodic_strength)
    aperiodic_weight = torch.sqrt((1.0 - periodic_strength).clamp_min(0.0))
    residual = periodic_wave * periodic_weight + aperiodic_wave * aperiodic_weight
    if not bool(torch.isfinite(residual).all()):
        raise RuntimeError("unified residual synthesis produced non-finite samples")
    return residual.contiguous(), periodic_wave.contiguous(), aperiodic_wave.contiguous(), periodic_strength


def _relative_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
    return ((prediction - target).abs() / scale).mean()


def _weighted_relative_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("weighted loss geometry mismatch")
    denominator = weight.sum(dim=-1).clamp_min(1.0)
    target_scale = ((target.abs() * weight).sum(dim=-1) / denominator).clamp_min(1.0e-4)
    error = ((prediction - target).abs() * weight).sum(dim=-1) / denominator
    return (error / target_scale).mean()


def _weighted_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != weight.shape:
        raise ValueError("weighted smooth-L1 geometry mismatch")
    raw = F.smooth_l1_loss(prediction, target, reduction="none") * weight
    return raw.sum() / weight.sum().clamp_min(1.0)


def _target_vector_log_rms(target_vectors: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(target_vectors.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))
    return torch.log(rms)


def _sequence_log_rms_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(torch.log(_rms(prediction)), torch.log(_rms(target)))


def _derivative_relative_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_d = prediction[:, 1:] - prediction[:, :-1]
    target_d = target[:, 1:] - target[:, :-1]
    scale = target_d.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-4)
    return ((pred_d - target_d).abs() / scale).mean()


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
        losses.append(F.l1_loss(torch.log(pred.clamp_min(1.0e-5)), torch.log(ref.clamp_min(1.0e-5))))
    return torch.stack(losses).mean()


def _copy_pitch_sync_context(root: Path, model: LykenoxUnifiedPhaseResidualSourceV1) -> str | None:
    checkpoint = root / "models" / "lykenox_identity" / "training" / "pitch_synchronous_cycle_source_v1" / "best.pt"
    if not checkpoint.exists():
        return None
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != PITCH_SYNC_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("pitch-sync warm-start checkpoint schema mismatch")
    own = model.state_dict()
    copied = 0
    for key, value in payload["model_state"].items():
        if not (key.startswith("conditioning_projection.") or key.startswith("context_blocks.")):
            continue
        if key in own and own[key].shape == value.shape:
            own[key] = value.detach().clone()
            copied += 1
    if copied < 2:
        raise RuntimeError("pitch-sync warm start copied no meaningful context parameters")
    model.load_state_dict(own)
    return str(checkpoint)


def _loss_terms(
    model: LykenoxUnifiedPhaseResidualSourceV1,
    tensors: tuple[torch.Tensor, ...],
    *,
    phase_offset_cycles: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mel, f0_hz, voiced, periodicity, target_vectors, target_residual, oracle_cepstrum, reference = tensors
    harmonic_pairs, periodic_log_rms, aperiodic_vectors, aperiodic_log_rms = model(
        mel, f0_hz, voiced, periodicity
    )
    phase_offset = torch.full(
        (int(mel.shape[0]),), float(phase_offset_cycles), dtype=mel.dtype, device=mel.device
    )
    residual, periodic_wave, aperiodic_wave, strength_sample = synthesize_unified_residual(
        harmonic_pairs,
        aperiodic_vectors,
        f0_hz,
        voiced,
        periodicity,
        phase_offset_cycles=phase_offset,
    )
    prediction = render_time_varying_minimum_phase(
        residual, oracle_cepstrum, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    if residual.shape != target_residual.shape or prediction.shape != reference.shape:
        raise RuntimeError("unified source output geometry changed")

    # Direct component authority only where the conditioning makes that component identifiable.
    stable_periodic = strength_sample.square()
    stable_aperiodic = (1.0 - strength_sample).square()
    periodic_direct = _weighted_relative_l1(periodic_wave, target_residual, stable_periodic)
    aperiodic_direct = _weighted_relative_l1(aperiodic_wave, target_residual, stable_aperiodic)

    target_log_rms = _target_vector_log_rms(target_vectors)
    frame_strength = (voiced.clamp(0.0, 1.0) * periodicity.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    periodic_level = _weighted_smooth_l1(
        periodic_log_rms,
        target_log_rms[:, :-1],
        frame_strength.square(),
    )
    terminal_strength = frame_strength[:, -1:]
    extended_strength = torch.cat((frame_strength, terminal_strength), dim=1)
    aperiodic_level = _weighted_smooth_l1(
        aperiodic_log_rms,
        target_log_rms,
        (1.0 - extended_strength).square(),
    )

    residual_l1 = _relative_l1(residual, target_residual)
    residual_level = _sequence_log_rms_loss(residual, target_residual)
    residual_derivative = _derivative_relative_l1(residual, target_residual)
    waveform_l1 = _relative_l1(prediction, reference)
    waveform_level = _sequence_log_rms_loss(prediction, reference)
    spectral = _true_log_stft_loss(prediction, reference)

    total = (
        0.75 * periodic_direct
        + 0.75 * aperiodic_direct
        + 0.50 * periodic_level
        + 0.50 * aperiodic_level
        + 1.00 * residual_l1
        + 0.75 * residual_level
        + 0.50 * residual_derivative
        + 0.75 * waveform_l1
        + 0.75 * waveform_level
        + 0.50 * spectral
    )
    public = {
        "total": float(total.detach()),
        "periodic_direct": float(periodic_direct.detach()),
        "aperiodic_direct": float(aperiodic_direct.detach()),
        "periodic_log_rms": float(periodic_level.detach()),
        "aperiodic_log_rms": float(aperiodic_level.detach()),
        "residual_relative_l1": float(residual_l1.detach()),
        "residual_log_rms": float(residual_level.detach()),
        "residual_derivative": float(residual_derivative.detach()),
        "waveform_relative_l1": float(waveform_l1.detach()),
        "waveform_log_rms": float(waveform_level.detach()),
        "waveform_true_log_stft": float(spectral.detach()),
        "residual_rms_ratio": float((_rms(residual) / _rms(target_residual)).mean().detach()),
        "waveform_rms_ratio": float((_rms(prediction) / _rms(reference)).mean().detach()),
    }
    return total, public


@torch.no_grad()
def _evaluate_complete(
    model: LykenoxUnifiedPhaseResidualSourceV1,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for utterance in utterances:
        target = _load_or_build_target(root, utterance)
        tensors = _segment(utterance, target, start=0, frames=int(utterance.mel_frames))
        _, terms = _loss_terms(model, tensors, phase_offset_cycles=0.0)
        totals.append(terms)
    keys = tuple(totals[0])
    return {key: sum(item[key] for item in totals) / float(len(totals)) for key in keys}


def _save_checkpoint(
    path: Path,
    model: LykenoxUnifiedPhaseResidualSourceV1,
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
        "architecture": UNIFIED_PHASE_SOURCE_ARCHITECTURE,
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


def train_unified_phase_residual_source_v1(
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
    if train_items < 1 or val_items < 1 or segment_frames < 32 or max_updates < 1:
        raise ValueError("invalid unified-source training limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / "unified_phase_residual_source_v1"
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)

    model = LykenoxUnifiedPhaseResidualSourceV1().cpu()
    pitch_sync_context_warm_start = _copy_pitch_sync_context(root, model)
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
        "harmonic_count": HARMONIC_COUNT,
        "single_model": True,
        "single_recurrent_state": True,
        "teacher_forcing": False,
        "second_source_checkpoint_fallback": False,
        "source_handoff_or_bridge": False,
        "complementary_periodic_aperiodic_energy": True,
        "complete_heldout_validation": True,
        "pitch_sync_context_warm_start": pitch_sync_context_warm_start,
    }

    start_update = 0
    best_val = math.inf
    if resume and latest.exists():
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(latest, map_location="cpu")
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("unified-source checkpoint schema mismatch")
        if payload.get("architecture") != UNIFIED_PHASE_SOURCE_ARCHITECTURE:
            raise RuntimeError("unified-source checkpoint architecture mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_update = int(payload["update"])
        best_val = float(payload.get("best_val_total", math.inf))

    run_dir.mkdir(parents=True, exist_ok=True)
    history = run_dir / "history.jsonl"
    for update in range(start_update + 1, max_updates + 1):
        model.train()
        selected = (int(seed) + update * 7919) % len(train_set)
        utterance = train_set[selected]
        target = _load_or_build_target(root, utterance)
        frames = min(int(segment_frames), int(utterance.mel_frames))
        start = _deterministic_crop(
            int(utterance.mel_frames), frames, update=update, seed=seed + selected
        )
        tensors = _segment(utterance, target, start=start, frames=frames)
        phase_offset = accumulated_phase_offset(utterance, start)

        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss_terms(model, tensors, phase_offset_cycles=phase_offset)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("unified-source gradient norm became non-finite")
        optimizer.step()
        record: dict[str, object] = {
            "update": update,
            "utterance_id": utterance.utterance_id,
            "start_frame": start,
            "frames": frames,
            "teacher_forcing": False,
            "grad_norm": grad_norm,
            **terms,
        }
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = _evaluate_complete(model, root, val_set)
            record["validation"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(best, model, optimizer, update=update, best_val=best_val, config=config)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(latest, model, optimizer, update=update, best_val=best_val, config=config)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "unified_phase_residual_source_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": UNIFIED_PHASE_SOURCE_ARCHITECTURE,
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "pitch_sync_context_warm_start": pitch_sync_context_warm_start,
        "single_model": True,
        "single_recurrent_state": True,
        "codebook_used": False,
        "teacher_forcing_used": False,
        "second_source_checkpoint_fallback_used": False,
        "source_handoff_or_bridge_used": False,
        "stochastic_innovation_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_complete_heldout_unified_source_and_listen_against_pitch_sync_and_identity_ceiling",
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
    print(json.dumps(train_unified_phase_residual_source_v1(
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
