"""CPU-first trainer for the LYKENOX coherent + innovation residual source.

The held-out V2 source recovered correct level and pronunciation and removed gangoso/chillido, but
remained perceptually robotic.  V2 was deterministic and therefore regressed stochastic/aperiodic
residual fine structure toward a deterministic mean.  This trainer keeps the V2 coherent and level
paths, warm-starts them from the owned V2 checkpoint when available, and trains a separate stochastic
innovation amount/color from owned Step-3f residual targets.

Exact sample losses are intentionally weak where stochastic phase is irreducible.  The innovation
amount is supervised by pitch-lag repeatability of the owned residual; residual/vector spectral
energy and final waveform magnitude/level remain directly supervised.  Validation is complete,
free-running held-out audio.  Metrics can reject but cannot accept product quality.
Policy: LYX-POL-001. CPU only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lykenox_voice_engine.models.vocoder.network_minimum_phase_coherent_innovation_source_v1 import (
    COHERENT_INNOVATION_ARCHITECTURE,
    HOP_LENGTH,
    INNOVATION_BANDS,
    RESIDUAL_VECTOR_SAMPLES,
    LykenoxCoherentInnovationResidualSourceV1,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v1 import (
    _load_or_build_target,
    _ola_vectors,
    _segment,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_train_v2 import (
    _relative_l1,
    _sequence_log_rms_loss,
    _target_vector_log_rms,
    _true_log_stft_loss,
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


TRAINER_VERSION = "owned-coherent-innovation-residual-source-trainer-v1"
CHECKPOINT_SCHEMA_VERSION = "owned-coherent-innovation-residual-source-checkpoint-v1"
POLICY_ID = "LYX-POL-001"
DEFAULT_TRAIN_ITEMS = 48
DEFAULT_VAL_ITEMS = 3
DEFAULT_SEGMENT_FRAMES = 96
DEFAULT_MAX_UPDATES = 600
DEFAULT_LEARNING_RATE = 1.5e-4
DEFAULT_WEIGHT_DECAY = 1.0e-5
DEFAULT_GRAD_CLIP = 5.0
DEFAULT_EVAL_EVERY = 50
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_SEED = 20260917
EPSILON = 1.0e-7


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _stable_seed(text: str, *, base: int) -> int:
    digest = hashlib.sha256(f"{base}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _unit_rms(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + EPSILON * EPSILON)


def _pitch_lag_aperiodicity_target(
    target_vectors: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    periodicity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (innovation_fraction, coherence) from owned residual repeatability.

    For voiced frames, repeatability at the local pitch lag is measured directly on the target
    residual vector.  Cached periodicity is used only as a stabilizing prior.  Unvoiced frames are
    treated as innovation-dominant.  No learned probe or external model is involved.
    """
    if target_vectors.ndim != 3 or target_vectors.shape[-1] != RESIDUAL_VECTOR_SAMPLES:
        raise ValueError("target_vectors must be [B,T+1,512]")
    batch, vector_count, _ = target_vectors.shape
    frames = int(f0_hz.shape[1])
    innovation = torch.zeros(batch, vector_count, dtype=target_vectors.dtype)
    coherence_out = torch.zeros_like(innovation)
    with torch.no_grad():
        for b in range(batch):
            for t in range(vector_count):
                c = min(t, frames - 1)
                v = float(voiced[b, c])
                f0 = float(f0_hz[b, c])
                prior = max(0.0, min(1.0, v * float(periodicity[b, c])))
                if v < 0.25 or not math.isfinite(f0) or f0 <= 1.0:
                    coherence = 0.0
                else:
                    lag = int(round(float(SAMPLE_RATE) / f0))
                    x = target_vectors[b, t].to(torch.float64)
                    if lag < 8 or lag >= RESIDUAL_VECTOR_SAMPLES - 16:
                        coherence = prior
                    else:
                        left = x[:-lag] - x[:-lag].mean()
                        right = x[lag:] - x[lag:].mean()
                        denom = torch.sqrt(
                            left.square().sum().clamp_min(1.0e-12)
                            * right.square().sum().clamp_min(1.0e-12)
                        )
                        rho = float((left * right).sum().abs() / denom)
                        rho = max(0.0, min(1.0, rho))
                        coherence = max(0.0, min(1.0, 0.75 * rho + 0.25 * prior))
                # Energy-preserving mix uses sqrt(1-coherence^2) as the innovation coordinate.
                mix = math.sqrt(max(0.0, 1.0 - coherence * coherence))
                innovation[b, t] = mix
                coherence_out[b, t] = coherence
    return innovation, coherence_out


def _coherent_shape_loss(
    coherent_shape: torch.Tensor,
    target_vectors: torch.Tensor,
    coherence: torch.Tensor,
) -> torch.Tensor:
    target_shape = _unit_rms(target_vectors)
    pred_norm = torch.sqrt(coherent_shape.square().sum(dim=-1).clamp_min(1.0e-10))
    target_norm = torch.sqrt(target_shape.square().sum(dim=-1).clamp_min(1.0e-10))
    cosine = (coherent_shape * target_shape).sum(dim=-1) / (pred_norm * target_norm)
    per_vector = 1.0 - cosine.clamp(-1.0, 1.0)
    weights = coherence.detach().clamp(0.0, 1.0)
    return (per_vector * weights).sum() / weights.sum().clamp_min(1.0)


def _vector_band_log_energy(vectors: torch.Tensor, bands: int = INNOVATION_BANDS) -> torch.Tensor:
    spectrum = torch.fft.rfft(vectors, n=RESIDUAL_VECTOR_SAMPLES, dim=-1)
    power = spectrum.real.square() + spectrum.imag.square()
    bins = int(power.shape[-1])
    edges = torch.linspace(0, bins, bands + 1, device=power.device).round().to(torch.int64)
    values: list[torch.Tensor] = []
    for band in range(bands):
        left = int(edges[band])
        right = max(left + 1, int(edges[band + 1]))
        right = min(right, bins)
        values.append(torch.log(power[..., left:right].mean(dim=-1).clamp_min(1.0e-8)))
    return torch.stack(values, dim=-1)


def _vector_spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(_vector_band_log_energy(prediction), _vector_band_log_energy(target))


def _loss_terms(
    model: LykenoxCoherentInnovationResidualSourceV1,
    tensors: tuple[torch.Tensor, ...],
    *,
    innovation_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    mel, f0_hz, voiced, periodicity, target_vectors, target_residual, oracle_cepstrum, reference = tensors
    predicted_vectors, coherent_shape, _, predicted_log_rms, predicted_mix = model.forward_with_components(
        mel,
        f0_hz,
        voiced,
        periodicity,
        innovation_seed=int(innovation_seed),
    )
    target_log_rms = _target_vector_log_rms(target_vectors)
    target_mix, coherence = _pitch_lag_aperiodicity_target(
        target_vectors, f0_hz, voiced, periodicity
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
        raise RuntimeError("coherent-innovation renderer output geometry changed")

    coherent_shape_loss = _coherent_shape_loss(coherent_shape, target_vectors, coherence)
    innovation_mix_loss = F.smooth_l1_loss(predicted_mix, target_mix)
    vector_level_loss = F.smooth_l1_loss(predicted_log_rms, target_log_rms)
    vector_spectral_loss = _vector_spectral_loss(predicted_vectors, target_vectors)
    residual_level_loss = _sequence_log_rms_loss(predicted_residual, target_residual)
    waveform_level_loss = _sequence_log_rms_loss(prediction, reference)
    residual_spectral_loss = _true_log_stft_loss(predicted_residual, target_residual)
    waveform_spectral_loss = _true_log_stft_loss(prediction, reference)

    # Exact waveform phase is not a valid target for stochastic innovation, so samplewise errors
    # are intentionally secondary.  They still anchor the coherent path without driving innovation
    # toward zero merely because a different random realization is used.
    residual_l1 = _relative_l1(predicted_residual, target_residual)
    waveform_l1 = _relative_l1(prediction, reference)

    total = (
        0.80 * coherent_shape_loss
        + 1.50 * innovation_mix_loss
        + 1.25 * vector_level_loss
        + 0.90 * vector_spectral_loss
        + 1.00 * residual_level_loss
        + 1.25 * waveform_level_loss
        + 0.75 * residual_spectral_loss
        + 0.75 * waveform_spectral_loss
        + 0.15 * residual_l1
        + 0.15 * waveform_l1
    )
    pred_rms = torch.sqrt(prediction.square().mean(dim=-1).clamp_min(1.0e-14))
    ref_rms = torch.sqrt(reference.square().mean(dim=-1).clamp_min(1.0e-14))
    public = {
        "total": float(total.detach()),
        "coherent_shape": float(coherent_shape_loss.detach()),
        "innovation_mix": float(innovation_mix_loss.detach()),
        "predicted_innovation_mix_mean": float(predicted_mix.mean().detach()),
        "target_innovation_mix_mean": float(target_mix.mean().detach()),
        "vector_log_rms": float(vector_level_loss.detach()),
        "vector_band_log_energy": float(vector_spectral_loss.detach()),
        "residual_log_rms": float(residual_level_loss.detach()),
        "waveform_log_rms": float(waveform_level_loss.detach()),
        "residual_log_stft": float(residual_spectral_loss.detach()),
        "waveform_log_stft": float(waveform_spectral_loss.detach()),
        "waveform_rms_ratio": float((pred_rms / ref_rms).mean().detach()),
    }
    return total, public


@torch.no_grad()
def _evaluate_complete(
    model: LykenoxCoherentInnovationResidualSourceV1,
    root: Path,
    utterances: list[OwnedVocoderUtterance],
    *,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: list[dict[str, float]] = []
    for utterance in utterances:
        target = _load_or_build_target(root, utterance)
        tensors = _segment(
            utterance, target, start=0, frames=int(utterance.mel_frames)
        )
        _, terms = _loss_terms(
            model,
            tensors,
            innovation_seed=_stable_seed(utterance.utterance_id, base=seed),
        )
        totals.append(terms)
    keys = tuple(totals[0])
    return {key: sum(item[key] for item in totals) / float(len(totals)) for key in keys}


def _warm_start_from_v2(root: Path, model: LykenoxCoherentInnovationResidualSourceV1) -> dict[str, object]:
    path = root / "models" / "lykenox_identity" / "training" / "continuous_residual_source_v2" / "best.pt"
    if not path.exists():
        return {"used": False, "path": str(path), "loaded_tensor_count": 0}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    source_state = payload.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError("V2 warm-start checkpoint has no model_state")
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    target_state.update(compatible)
    model.load_state_dict(target_state)
    return {
        "used": True,
        "path": str(path),
        "source_update": int(payload.get("update", -1)),
        "loaded_tensor_count": len(compatible),
    }


def _save_checkpoint(
    path: Path,
    model: LykenoxCoherentInnovationResidualSourceV1,
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
        "architecture": COHERENT_INNOVATION_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
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


def train_coherent_innovation_source(
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
        raise ValueError("invalid coherent-innovation training limits")
    root = Path(root).resolve()
    torch.manual_seed(int(seed))
    run_dir = root / "models" / "lykenox_identity" / "training" / "coherent_innovation_source_v1"
    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    model = LykenoxCoherentInnovationResidualSourceV1().cpu()
    warm_start: dict[str, object] = {"used": False, "loaded_tensor_count": 0}
    start_update = 0
    best_val = math.inf

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=DEFAULT_WEIGHT_DECAY
    )
    if resume and latest.exists():
        try:
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest, map_location="cpu")
        if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("coherent-innovation checkpoint schema mismatch")
        if checkpoint.get("architecture") != COHERENT_INNOVATION_ARCHITECTURE:
            raise RuntimeError("coherent-innovation checkpoint architecture mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_update = int(checkpoint["update"])
        best_val = float(checkpoint.get("best_val_total", math.inf))
        warm_start = dict(checkpoint.get("warm_start", warm_start))
    else:
        warm_start = _warm_start_from_v2(root, model)

    train_set = collect_owned_vocoder_utterances(root, "train", max_items=train_items)
    val_set = collect_owned_vocoder_utterances(root, "val", max_items=val_items)
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
        "innovation_bands": INNOVATION_BANDS,
        "codebook_used": False,
        "complete_heldout_validation": True,
        "teacher_forcing_used": False,
        "v2_warm_start_allowed": True,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    for update in range(start_update + 1, max_updates + 1):
        model.train()
        utterance_index = (int(seed) + update * 7919) % len(train_set)
        utterance = train_set[utterance_index]
        target = _load_or_build_target(root, utterance)
        frames = min(segment_frames, int(utterance.mel_frames))
        if int(utterance.mel_frames) <= frames:
            start = 0
        else:
            span = int(utterance.mel_frames) - frames
            start = (int(seed) * 1000003 + update * 9176 + 37) % (span + 1)
        tensors = _segment(utterance, target, start=int(start), frames=frames)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss_terms(
            model,
            tensors,
            innovation_seed=_stable_seed(
                f"{utterance.utterance_id}:{update}", base=seed
            ),
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise RuntimeError("coherent-innovation gradient norm became non-finite")
        optimizer.step()

        record: dict[str, object] = {
            "update": update,
            "utterance_id": utterance.utterance_id,
            "start_frame": int(start),
            "grad_norm": grad_norm,
            **terms,
        }
        if update % DEFAULT_EVAL_EVERY == 0 or update == max_updates:
            validation = _evaluate_complete(model, root, val_set, seed=seed + 100000)
            record["validation"] = validation
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
        with history_path.open("a", encoding="utf-8") as handle:
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
        "status": "coherent_innovation_source_training_complete",
        "policy_id": POLICY_ID,
        "trainer_version": TRAINER_VERSION,
        "architecture": COHERENT_INNOVATION_ARCHITECTURE,
        "renderer_version": RENDERER_VERSION,
        "device": "cpu",
        "updates": max_updates,
        "best_val_total": best_val,
        "best_checkpoint": str(best),
        "latest_checkpoint": str(latest),
        "warm_start": warm_start,
        "training_split_only_for_optimizer_updates": True,
        "codebook_used": False,
        "teacher_forcing_used": False,
        "third_party_model_or_weight_used": False,
        "remote_service_used": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "production_accepted_by_metrics": False,
        "next_action": "render_complete_heldout_and_listen_for_robotic_timbre_reduction",
    }
    _atomic_json(run_dir / "training_report.json", report)
    return report


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_UPDATES",
    "DEFAULT_SEGMENT_FRAMES",
    "DEFAULT_SEED",
    "DEFAULT_TRAIN_ITEMS",
    "DEFAULT_VAL_ITEMS",
    "POLICY_ID",
    "TRAINER_VERSION",
    "train_coherent_innovation_source",
]
