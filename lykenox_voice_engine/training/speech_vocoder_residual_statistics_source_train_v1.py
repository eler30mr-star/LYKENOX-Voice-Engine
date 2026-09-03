"""Train the LYKENOX residual-statistics source without deterministic waveform regression.

Forensics proved that deterministic 512/256 and 256-sample frame heads repeat a learned waveform
prototype at the 93.75 Hz acoustic hop.  This trainer therefore never asks the model to predict
residual phase samples.  It supervises only identifiable Step-3f residual statistics: source
cepstral spectral shape, explicit frame RMS, residual periodicity, multi-resolution log magnitude and
high-band spectral flatness.

A continuous full-utterance source carrier (continuous F0 phase + one absolute-index deterministic
noise stream) is shaped by the predicted source cepstrum.  Neither pulse phase nor noise phase resets
at frame boundaries.  The carrier is part of source generation, not post-hoc innovation mixing.
No external model/weight/service, codebook, gain normalization, EQ, denoise or duration modification
is used. Policy: LYX-POL-001. CPU is the reference device.
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

from lykenox_voice_engine.models.vocoder.network_minimum_phase_residual_statistics_source_v1 import (
    RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE,
    SOURCE_CEPSTRAL_ORDER,
    LykenoxResidualStatisticsSourceV1,
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
from lykenox_voice_engine.training.speech_vocoder_continuous_stream_source_train_v1 import (
    CHECKPOINT_SCHEMA_VERSION as STREAM_CHECKPOINT_SCHEMA,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    N_FFT,
    RENDERER_VERSION,
    SAMPLE_RATE,
    fixed_linear_frame_to_sample,
    one_sided_real_cepstrum_to_minimum_phase_fir,
    render_time_varying_minimum_phase,
)


TRAINER_VERSION = "owned-residual-statistics-source-trainer-v1"
CHECKPOINT_SCHEMA_VERSION = "owned-residual-statistics-source-checkpoint-v1"
POLICY_ID = "LYX-POL-001"
RUN_DIR_NAME = "residual_statistics_source_v1"
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 3
DEFAULT_SEGMENT_FRAMES = 96
DEFAULT_MAX_UPDATES = 600
DEFAULT_LEARNING_RATE = 2.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 50
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_SEED = 20261003
TARGET_FRAME_LENGTH = N_FFT
EPSILON = 1.0e-7
HIGH_BAND_LOW_HZ = 3500.0
HIGH_BAND_HIGH_HZ = 11000.0


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


def _utterance_seed(utterance_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(utterance_id.encode("utf-8")).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "little") % 1000003


def _centered_frames(residual: torch.Tensor, frame_count: int) -> torch.Tensor:
    half = TARGET_FRAME_LENGTH // 2
    padded = F.pad(residual.view(1, 1, -1), (half, half), mode="reflect")[0, 0]
    frames = padded.unfold(0, TARGET_FRAME_LENGTH, HOP_LENGTH)
    if int(frames.shape[0]) < frame_count:
        raise RuntimeError("residual statistics framing produced too few frames")
    return frames[:frame_count].to(torch.float32).contiguous()


def _target_statistics(
    residual: torch.Tensor,
    conditioning: PitchConditioningV2,
    *,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frames = _centered_frames(residual, frame_count)
    window = torch.hann_window(TARGET_FRAME_LENGTH, periodic=True, dtype=frames.dtype)
    windowed = frames * window.unsqueeze(0)
    rms = torch.sqrt(windowed.square().mean(dim=-1).clamp_min(EPSILON * EPSILON))
    log_rms = torch.log(rms)

    magnitude = torch.fft.rfft(windowed, n=N_FFT, dim=-1).abs().clamp_min(1.0e-6)
    log_magnitude = torch.log(magnitude)
    centered_log_magnitude = log_magnitude - log_magnitude.mean(dim=-1, keepdim=True)
    full_cepstrum = torch.fft.irfft(centered_log_magnitude, n=N_FFT, dim=-1)
    source_cepstrum = full_cepstrum[:, :SOURCE_CEPSTRAL_ORDER].contiguous()
    source_cepstrum[:, 0] = 0.0

    periodicity_rows: list[torch.Tensor] = []
    for index in range(frame_count):
        f0 = float(conditioning.f0_track_hz[index])
        if not math.isfinite(f0) or f0 <= 1.0:
            periodicity_rows.append(torch.zeros((), dtype=frames.dtype))
            continue
        lag = max(1, min(TARGET_FRAME_LENGTH - 8, int(round(SAMPLE_RATE / f0))))
        value = windowed[index] - windowed[index].mean()
        left = value[:-lag]
        right = value[lag:]
        denominator = torch.sqrt(left.square().sum() * right.square().sum()).clamp_min(1.0e-10)
        correlation = (left * right).sum() / denominator
        periodicity_rows.append(correlation.clamp(0.0, 1.0))
    residual_periodicity = torch.stack(periodicity_rows).to(torch.float32).contiguous()
    return source_cepstrum, log_rms.contiguous(), residual_periodicity


def _continuous_noise(sample_count: int, *, seed: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    index = torch.arange(sample_count, dtype=dtype, device=device)
    phase = (index + float(seed) * 131.0) * 12.9898 + 78.233
    hashed = torch.sin(phase) * 43758.5453123
    return (hashed - torch.floor(hashed)).mul(2.0).sub(1.0)


def _lowpass_pulse(pulse: torch.Tensor) -> torch.Tensor:
    taps = 63
    cutoff_hz = 10800.0
    center = (taps - 1) / 2.0
    n = torch.arange(taps, dtype=pulse.dtype, device=pulse.device) - center
    normalized_cutoff = cutoff_hz / float(SAMPLE_RATE)
    ideal = 2.0 * normalized_cutoff * torch.sinc(2.0 * normalized_cutoff * n)
    window = torch.hann_window(taps, periodic=False, dtype=pulse.dtype, device=pulse.device)
    kernel = ideal * window
    kernel = kernel / kernel.sum().clamp_min(EPSILON)
    padding = (taps - 1) // 2
    return F.conv1d(pulse.view(1, 1, -1), kernel.view(1, 1, -1), padding=padding)[0, 0]


def _continuous_carrier(
    f0_track_hz: torch.Tensor,
    residual_periodicity: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    if f0_track_hz.ndim != 1 or residual_periodicity.shape != f0_track_hz.shape:
        raise ValueError("carrier conditioning must be one frame sequence")
    f0 = fixed_linear_frame_to_sample(f0_track_hz.unsqueeze(0), hop_length=HOP_LENGTH)[0]
    mix = fixed_linear_frame_to_sample(residual_periodicity.clamp(0.0, 1.0).unsqueeze(0), hop_length=HOP_LENGTH)[0]
    phase_increment = torch.where(f0 > 1.0, f0 / float(SAMPLE_RATE), torch.zeros_like(f0))
    accumulated = torch.cumsum(phase_increment, dim=0)
    previous = F.pad(accumulated[:-1], (1, 0), value=0.0)
    pulse = (torch.floor(accumulated) > torch.floor(previous)).to(f0.dtype)
    pulse_scale = torch.where(
        f0 > 1.0,
        torch.sqrt(float(SAMPLE_RATE) / f0.clamp_min(1.0)),
        torch.zeros_like(f0),
    )
    pulse = _lowpass_pulse(pulse * pulse_scale)
    noise = _continuous_noise(int(f0.numel()), seed=seed, dtype=f0.dtype, device=f0.device)
    periodic_weight = torch.sqrt(mix.clamp(0.0, 1.0))
    aperiodic_weight = torch.sqrt((1.0 - mix).clamp(0.0, 1.0))
    carrier = periodic_weight * pulse + aperiodic_weight * noise
    carrier_rms = torch.sqrt(carrier.square().mean().clamp_min(EPSILON * EPSILON))
    return (carrier / carrier_rms).contiguous()


def _unit_energy_source_cepstrum(source_cepstrum: torch.Tensor) -> torch.Tensor:
    normalized = source_cepstrum.clone()
    normalized[..., 0] = 0.0
    impulse = one_sided_real_cepstrum_to_minimum_phase_fir(normalized, n_fft=N_FFT)
    l2 = torch.sqrt(impulse.square().sum(dim=-1).clamp_min(EPSILON * EPSILON))
    normalized[..., 0] = -torch.log(l2)
    return normalized


def synthesize_residual_from_statistics(
    source_cepstrum: torch.Tensor,
    log_rms: torch.Tensor,
    residual_periodicity: torch.Tensor,
    f0_track_hz: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    if source_cepstrum.ndim != 3 or source_cepstrum.shape[0] != 1:
        raise ValueError("source statistics synthesizer expects batch=1")
    if log_rms.shape != source_cepstrum.shape[:2] or residual_periodicity.shape != log_rms.shape or f0_track_hz.shape != log_rms.shape:
        raise ValueError("source statistics geometry mismatch")
    carrier = _continuous_carrier(f0_track_hz[0], residual_periodicity[0], seed=seed)
    normalized_cepstrum = _unit_energy_source_cepstrum(source_cepstrum)
    shaped = render_time_varying_minimum_phase(
        carrier.unsqueeze(0), normalized_cepstrum, hop_length=HOP_LENGTH, n_fft=N_FFT
    )[0]
    gain = fixed_linear_frame_to_sample(torch.exp(log_rms).unsqueeze(0) if log_rms.ndim == 1 else torch.exp(log_rms), hop_length=HOP_LENGTH)
    if gain.ndim == 2:
        gain = gain[0]
    residual = shaped * gain
    if not bool(torch.isfinite(residual).all()):
        raise RuntimeError("statistics source synthesizer produced non-finite residual")
    return residual.contiguous()


def _log_spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(n_fft, dtype=prediction.dtype, device=prediction.device)
        pred = torch.stft(prediction, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True).abs()
        ref = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True).abs()
        losses.append(F.l1_loss(torch.log(pred.clamp_min(1.0e-5)), torch.log(ref.clamp_min(1.0e-5))))
    return torch.stack(losses).mean()


def _high_band_flatness(value: torch.Tensor) -> torch.Tensor:
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft, dtype=value.dtype, device=value.device)
    magnitude = torch.stft(value, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True).abs().clamp_min(1.0e-6)
    frequencies = torch.fft.rfftfreq(n_fft, 1.0 / float(SAMPLE_RATE)).to(value.device)
    mask = (frequencies >= HIGH_BAND_LOW_HZ) & (frequencies <= HIGH_BAND_HIGH_HZ)
    band = magnitude[..., mask]
    geometric = torch.exp(torch.log(band).mean(dim=-1))
    arithmetic = band.mean(dim=-1).clamp_min(1.0e-6)
    return (geometric / arithmetic).mean()


def _deterministic_crop(frame_count: int, segment_frames: int, *, update: int, seed: int) -> int:
    if frame_count <= segment_frames:
        return 0
    span = frame_count - segment_frames
    return int((int(seed) * 1000003 + int(update) * 9176 + 101) % (span + 1))


def _loss_terms(
    model: LykenoxResidualStatisticsSourceV1,
    utterance: OwnedVocoderUtterance,
    conditioning: PitchConditioningV2,
    target: dict[str, Any],
    *,
    start: int,
    frames: int,
    carrier_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    end = start + frames
    sample_start = start * HOP_LENGTH
    sample_end = end * HOP_LENGTH
    target_residual = target["residual"][sample_start:sample_end].to(torch.float32).contiguous()
    local_conditioning = PitchConditioningV2(
        f0_track_hz=conditioning.f0_track_hz[start:end],
        periodic_strength=conditioning.periodic_strength[start:end],
        energy_confidence=conditioning.energy_confidence[start:end],
        raw_periodicity=conditioning.raw_periodicity[start:end],
        anchor_voiced=conditioning.anchor_voiced[start:end],
        frame_rms=conditioning.frame_rms[start:end],
    )
    target_cepstrum, target_log_rms, target_periodicity = _target_statistics(
        target_residual, local_conditioning, frame_count=frames
    )
    predicted_cepstrum, predicted_log_rms, predicted_periodicity = model(
        utterance.mel[start:end].unsqueeze(0).cpu(),
        local_conditioning.f0_track_hz.unsqueeze(0).cpu(),
        local_conditioning.energy_confidence.unsqueeze(0).cpu(),
        local_conditioning.periodic_strength.unsqueeze(0).cpu(),
    )
    generated = synthesize_residual_from_statistics(
        predicted_cepstrum,
        predicted_log_rms,
        predicted_periodicity,
        local_conditioning.f0_track_hz.unsqueeze(0).cpu(),
        seed=carrier_seed,
    )
    cepstral = F.smooth_l1_loss(predicted_cepstrum[0, :, 1:], target_cepstrum[:, 1:])
    level = F.smooth_l1_loss(predicted_log_rms[0], target_log_rms)
    periodicity = F.smooth_l1_loss(predicted_periodicity[0], target_periodicity)
    spectral = _log_spectral_loss(generated, target_residual)
    generated_flatness = _high_band_flatness(generated)
    target_flatness = _high_band_flatness(target_residual)
    flatness = F.smooth_l1_loss(generated_flatness, target_flatness)
    total = 1.00 * cepstral + 1.25 * level + 0.75 * periodicity + 1.00 * spectral + 0.75 * flatness
    public = {
        "total": float(total.detach()),
        "source_cepstrum": float(cepstral.detach()),
        "source_log_rms": float(level.detach()),
        "source_periodicity": float(periodicity.detach()),
        "residual_log_spectrum": float(spectral.detach()),
        "high_band_flatness": float(flatness.detach()),
        "generated_high_band_flatness": float(generated_flatness.detach()),
        "target_high_band_flatness": float(target_flatness.detach()),
    }
    return total, public


def _warm_start_context(root: Path, model: LykenoxResidualStatisticsSourceV1) -> dict[str, object]:
    checkpoint = root / "models" / "lykenox_identity" / "training" / "continuous_residual_stream_source_v1" / "best.pt"
    result: dict[str, object] = {"used": False, "checkpoint": str(checkpoint), "loaded_tensor_count": 0}
    if not checkpoint.exists():
        return result
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("checkpoint_schema_version") != STREAM_CHECKPOINT_SCHEMA:
        raise RuntimeError("stream warm-start checkpoint schema mismatch")
    state = payload["model_state"]
    own = model.state_dict()
    copied = 0
    for key, value in state.items():
        if not (key.startswith("conditioning_projection.") or key.startswith("context_blocks.") or key.startswith("recurrent.")):
            continue
        if key in own and own[key].shape == value.shape:
            own[key] = value.detach().clone()
            copied += 1
    if copied:
        model.load_state_dict(own)
        result.update({"used": True, "loaded_tensor_count": copied})
    return result


def _save_checkpoint(
    path: Path,
    model: LykenoxResidualStatisticsSourceV1,
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
        "architecture": RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "renderer_version": RENDERER_VERSION,
        "source_representation": "frame_statistics_plus_continuous_absolute_phase_carrier",
        "deterministic_waveform_regression": False,
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


def train_residual_statistics_source_v1(
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
    root = Path(root).resolve()
    if train_items < 1 or val_items < 1 or segment_frames < 32 or max_updates < 1:
        raise ValueError("invalid residual-statistics training limits")
    torch.manual_seed(int(seed))
    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
    conditioning_cache = {item.utterance_id: _conditioning(item) for item in train_set + val_set}
    model = LykenoxResidualStatisticsSourceV1().cpu()
    warm_start = _warm_start_context(root, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=DEFAULT_WEIGHT_DECAY)
    run_dir = root / "models" / "lykenox_identity" / "training" / RUN_DIR_NAME
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    config: dict[str, object] = {
        "train_items": train_items,
        "val_items": val_items,
        "segment_frames": segment_frames,
        "max_updates": max_updates,
        "learning_rate": learning_rate,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "source_cepstral_order": SOURCE_CEPSTRAL_ORDER,
        "conditioning_contract": PITCH_CONDITIONING_V2,
        "deterministic_waveform_regression": False,
        "carrier_phase_resets_per_frame": False,
        "carrier_noise_resets_per_frame": False,
        "train_split_only_for_optimizer_updates": True,
    }
    start_update = 0
    best_val = math.inf
    if resume and latest.exists():
        try:
            payload = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(latest, map_location="cpu")
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("residual-statistics checkpoint schema mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_update = int(payload["update"])
        best_val = float(payload.get("best_val_total", math.inf))

    history = run_dir / "history.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_complete() -> dict[str, float]:
        model.eval()
        rows: list[dict[str, float]] = []
        for utterance in val_set:
            target = _load_or_build_target(root, utterance)
            conditioning = conditioning_cache[utterance.utterance_id]
            _, public = _loss_terms(
                model,
                utterance,
                conditioning,
                target,
                start=0,
                frames=int(utterance.mel_frames),
                carrier_seed=_utterance_seed(utterance.utterance_id, seed + 900000),
            )
            rows.append(public)
        keys = tuple(rows[0])
        return {key: sum(row[key] for row in rows) / float(len(rows)) for key in keys}

    for update in range(start_update + 1, max_updates + 1):
        model.train()
        utterance_index = (int(seed) + update * 7919) % len(train_set)
        utterance = train_set[utterance_index]
        target = _load_or_build_target(root, utterance)
        conditioning = conditioning_cache[utterance.utterance_id]
        frames = min(segment_frames, int(utterance.mel_frames))
        start = _deterministic_crop(int(utterance.mel_frames), frames, update=update, seed=seed + utterance_index)
        optimizer.zero_grad(set_to_none=True)
        loss, public = _loss_terms(
            model,
            utterance,
            conditioning,
            target,
            start=start,
            frames=frames,
            carrier_seed=_utterance_seed(utterance.utterance_id, seed),
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("residual-statistics gradient norm became non-finite")
        optimizer.step()
        record: dict[str, object] = {"update": update, "utterance_id": utterance.utterance_id, "start_frame": start, "grad_norm": grad_norm, **public}
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = evaluate_complete()
            record["validation_complete_utterances"] = validation
            if validation["total"] < best_val:
                best_val = validation["total"]
                _save_checkpoint(best, model, optimizer, update=update, best_val=best_val, config=config, warm_start=warm_start)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if update % DEFAULT_CHECKPOINT_EVERY == 0 or update == max_updates:
            _save_checkpoint(latest, model, optimizer, update=update, best_val=best_val, config=config, warm_start=warm_start)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report: dict[str, object] = {
        "status": "residual_statistics_source_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": RESIDUAL_STATISTICS_SOURCE_ARCHITECTURE,
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "source_representation": "frame_statistics_plus_continuous_absolute_phase_carrier",
        "deterministic_waveform_regression": False,
        "carrier_phase_resets_per_frame": False,
        "carrier_noise_resets_per_frame": False,
        "teacher_forcing_used": False,
        "codebook_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_heldout_and_verify_hop_grid_signature_removed_before_human_listening",
    }
    _atomic_json(run_dir / "training_report.json", report)
    return report


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "RUN_DIR_NAME",
    "TRAINER_VERSION",
    "synthesize_residual_from_statistics",
    "train_residual_statistics_source_v1",
]


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
    print(json.dumps(train_residual_statistics_source_v1(
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
