"""Direct original-voice vs v4.2-oracle forensic audit.

V4.2 is driven only by target mel, target F0 and target voicing from the same held-out
utterance. Any remaining paired waveform gap is therefore inside the vocoder path, not
predicted duration or predicted acoustic conditioning. This audit is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V4_2_ARCHITECTURE
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import _wave_metrics
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import target_relative_presence_loss
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _load_reference_waveform,
)


AUDIT_VERSION = "vocoder-v4-2-reference-waveform-forensics-v1"
VALIDATION_INDICES = (0, 1, 2)
OUTPUT_DIR_NAME = "vocoder_v4_2_reference_waveform_forensics_v1"
WINDOW = 1024
PITCH_MIN_HZ = 60.0
PITCH_MAX_HZ = 400.0


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
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "detail_best": training / "acoustic_mel_detail_head_v1" / "best.pt",
        "postnet_best": training / "acoustic_mel_postnet_v1" / "best.pt",
        "mel_fidelity_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def _frames(wave: torch.Tensor, count: int, hop: int) -> torch.Tensor:
    padded = F.pad(wave, (WINDOW // 2, WINDOW // 2))
    frames = padded.unfold(0, WINDOW, hop)
    if frames.shape[0] < count:
        padded = F.pad(padded, (0, (count - frames.shape[0]) * hop))
        frames = padded.unfold(0, WINDOW, hop)
    return frames[:count].contiguous()


def _frame_spectrum(wave: torch.Tensor, count: int, sample_rate: int, hop: int):
    frames = _frames(wave, count, hop)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hann_window(WINDOW, dtype=frames.dtype)
    spec = torch.fft.rfft(frames * window, dim=-1)
    mag = spec.abs().clamp_min(1e-8)
    power = mag.square()
    freqs = torch.fft.rfftfreq(WINDOW, d=1.0 / sample_rate)
    return frames, mag, power, freqs


def _pitch_metrics(
    frames: torch.Tensor,
    target_f0: torch.Tensor,
    voiced: torch.Tensor,
    sample_rate: int,
) -> dict[str, float]:
    window = torch.hann_window(frames.shape[-1], dtype=frames.dtype)
    x = (frames - frames.mean(dim=-1, keepdim=True)) * window
    nfft = 2048
    spec = torch.fft.rfft(x, n=nfft, dim=-1)
    acf = torch.fft.irfft(spec.abs().square(), n=nfft, dim=-1)
    acf = acf / acf[:, :1].clamp_min(1e-8)
    min_lag = max(1, int(sample_rate / PITCH_MAX_HZ))
    max_lag = min(int(sample_rate / PITCH_MIN_HZ), acf.shape[-1] - 1)

    periodicity: list[float] = []
    cents: list[float] = []
    voiced_mask = voiced.bool() & torch.isfinite(target_f0) & (target_f0 > 0)
    for idx in torch.nonzero(voiced_mask, as_tuple=False).flatten().tolist():
        f0 = float(target_f0[idx])
        target_lag = sample_rate / max(f0, 1e-6)
        lo = max(min_lag, int(target_lag / 1.25))
        hi = min(max_lag, int(math.ceil(target_lag / 0.80)))
        if hi <= lo:
            continue
        values = acf[idx, lo : hi + 1]
        off = int(torch.argmax(values))
        lag = lo + off
        periodicity.append(float(values[off]))
        estimated = sample_rate / lag
        cents.append(abs(1200.0 * math.log2(estimated / f0)))

    unvoiced_mask = ~voiced.bool()
    uv_periodicity = 0.0
    if bool(unvoiced_mask.any()):
        broad = acf[unvoiced_mask, min_lag : max_lag + 1]
        uv_periodicity = float(broad.max(dim=-1).values.mean())
    return {
        "voiced_periodicity": sum(periodicity) / max(len(periodicity), 1),
        "voiced_pitch_mae_cents": sum(cents) / max(len(cents), 1),
        "unvoiced_periodicity": uv_periodicity,
    }


def _character_metrics(
    wave: torch.Tensor,
    target_f0: torch.Tensor,
    voiced: torch.Tensor,
    *,
    sample_rate: int,
    hop: int,
):
    count = int(target_f0.shape[0])
    frames, mag, power, freqs = _frame_spectrum(wave, count, sample_rate, hop)
    logmag = 20.0 * torch.log10(mag)
    flux = float((logmag[1:] - logmag[:-1]).abs().mean()) if count > 1 else 0.0
    env_band = (freqs >= 300.0) & (freqs <= 4000.0)
    smooth = F.avg_pool1d(logmag.unsqueeze(1), 9, stride=1, padding=4).squeeze(1)
    flat_band = (freqs >= 300.0) & (freqs <= 3000.0)
    p = power[:, flat_band].clamp_min(1e-12)
    flatness_db = 10.0 * torch.log10(
        (torch.exp(torch.log(p).mean(dim=-1)) / p.mean(dim=-1)).clamp_min(1e-12)
    )
    voiced_mask = voiced.bool()
    unvoiced_mask = ~voiced_mask
    high = (freqs >= 3000.0) & (freqs < 8000.0)
    uv_high = 0.0
    if bool(unvoiced_mask.any()):
        uv_power = power[unvoiced_mask]
        uv_high = float(uv_power[:, high].sum() / uv_power.sum().clamp_min(1e-12))
    result: dict[str, object] = {
        **_wave_metrics(wave, sample_rate),
        "spectral_flux_logmag_l1": flux,
        "voiced_spectral_flatness_db": (
            float(flatness_db[voiced_mask].mean()) if bool(voiced_mask.any()) else 0.0
        ),
        "unvoiced_3k_8k_fraction": uv_high,
        **_pitch_metrics(frames, target_f0, voiced, sample_rate),
    }
    return result, smooth, env_band


def _db10(value: float, reference: float) -> float:
    return 10.0 * math.log10(max(value, 1e-12) / max(reference, 1e-12))


def _pair_delta(
    ref: dict[str, object],
    gen: dict[str, object],
    ref_env: torch.Tensor,
    gen_env: torch.Tensor,
    env_band: torch.Tensor,
) -> dict[str, float]:
    ref_bands = dict(ref["band_power_fraction"])
    gen_bands = dict(gen["band_power_fraction"])
    return {
        "rms_relative_db": 20.0 * math.log10(max(float(gen["rms"]), 1e-12) / max(float(ref["rms"]), 1e-12)),
        "spectral_centroid_relative_pct": 100.0 * (float(gen["spectral_centroid_hz"]) / max(float(ref["spectral_centroid_hz"]), 1e-12) - 1.0),
        "band_80_300_fraction_relative_db": _db10(float(gen_bands["80_300"]), float(ref_bands["80_300"])),
        "band_300_3000_fraction_relative_db": _db10(float(gen_bands["300_3000"]), float(ref_bands["300_3000"])),
        "band_3000_nyquist_fraction_relative_db": _db10(float(gen_bands["3000_nyquist"]), float(ref_bands["3000_nyquist"])),
        "spectral_envelope_300_4k_l1_db": float((ref_env[:, env_band] - gen_env[:, env_band]).abs().mean()),
        "spectral_flux_relative_pct": 100.0 * (float(gen["spectral_flux_logmag_l1"]) / max(float(ref["spectral_flux_logmag_l1"]), 1e-12) - 1.0),
        "voiced_periodicity_delta": float(gen["voiced_periodicity"]) - float(ref["voiced_periodicity"]),
        "voiced_pitch_mae_cents_delta": float(gen["voiced_pitch_mae_cents"]) - float(ref["voiced_pitch_mae_cents"]),
        "unvoiced_periodicity_delta": float(gen["unvoiced_periodicity"]) - float(ref["unvoiced_periodicity"]),
        "voiced_spectral_flatness_delta_db": float(gen["voiced_spectral_flatness_db"]) - float(ref["voiced_spectral_flatness_db"]),
        "unvoiced_3k_8k_relative_db": _db10(float(gen["unvoiced_3k_8k_fraction"]), float(ref["unvoiced_3k_8k_fraction"])),
    }


def run_v4_2_reference_waveform_forensics(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    v4_2_path = protected["v4_2_best"]
    if not v4_2_path.exists():
        raise FileNotFoundError(f"accepted v4.2 checkpoint not found: {v4_2_path}")
    before = {name: _sha256(path) for name, path in protected.items()}

    config = LykenoxSpeechConfig()
    vocoder, _disc, payload = load_v4_2_checkpoint(v4_2_path)
    vocoder.cpu().eval()
    identity_exact = (
        vocoder.architecture == VOCODER_GENERATOR_V4_2_ARCHITECTURE
        and payload.get("generator_architecture") == VOCODER_GENERATOR_V4_2_ARCHITECTURE
    )
    if not identity_exact:
        raise RuntimeError("forensics requires accepted v4.2 architecture")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root, "val", config, duration_root=duration_root, include_pitch_targets=True
    )
    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reference_vs_v4_2_forensics.json"

    items: list[dict[str, object]] = []
    deltas: list[dict[str, float]] = []
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            raw = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("forensics requires target F0/voicing")
            frames = int(batch.mel_lengths[0])
            samples = frames * config.hop_length
            if int(batch.durations[0].sum()) != frames:
                raise RuntimeError("teacher duration grid mismatch")

            target_mel = batch.mel[:, :frames]
            target_f0 = batch.f0_hz[:, :frames]
            target_voiced = batch.voiced[:, :frames]
            generated_batch = vocoder(target_mel, target_f0, target_voiced)
            if tuple(generated_batch.shape) != (1, samples) or not bool(torch.isfinite(generated_batch).all()):
                raise RuntimeError("v4.2 oracle waveform structural failure")
            generated = generated_batch[0].detach().cpu().to(torch.float32).contiguous()
            reference = _load_reference_waveform(
                Path(str(raw["wav_path"])), sample_rate=config.sample_rate, samples=samples
            )

            ref_metrics, ref_env, env_band = _character_metrics(
                reference, target_f0[0], target_voiced[0],
                sample_rate=config.sample_rate, hop=config.hop_length
            )
            gen_metrics, gen_env, _ = _character_metrics(
                generated, target_f0[0], target_voiced[0],
                sample_rate=config.sample_rate, hop=config.hop_length
            )
            presence = target_relative_presence_loss(
                generated.unsqueeze(0), reference.unsqueeze(0),
                sample_rate=config.sample_rate, hop_length=config.hop_length
            )
            delta = _pair_delta(ref_metrics, gen_metrics, ref_env, gen_env, env_band)
            delta["presence_1k_8k_error_db"] = abs(float(presence.presence_1k_8k_error_db))
            deltas.append(delta)

            prefix = f"{audit_index:02d}"
            ref_path = output_dir / f"{prefix}_reference.wav"
            gen_path = output_dir / f"{prefix}_v4_2_oracle.wav"
            sf.write(str(ref_path), reference.numpy(), config.sample_rate, subtype="PCM_16")
            sf.write(str(gen_path), generated.numpy(), config.sample_rate, subtype="PCM_16")
            items.append({
                "audit_index": audit_index,
                "dataset_index": dataset_index,
                "utterance_id": str(item["utterance_id"]),
                "text": str(item["text"]),
                "conditioning": {
                    "target_mel": True,
                    "target_f0": True,
                    "target_voicing": True,
                    "teacher_duration_grid": True,
                },
                "reference": {"wav_path": str(ref_path), **ref_metrics},
                "v4_2_oracle": {"wav_path": str(gen_path), **gen_metrics},
                "v4_2_minus_reference": {k: round(float(v), 6) for k, v in delta.items()},
            })

    after = {name: _sha256(path) for name, path in protected.items()}
    unchanged = before == after
    keys = list(deltas[0]) if deltas else []
    means = {k: round(sum(d[k] for d in deltas) / len(deltas), 6) for k in keys} if deltas else {}
    report: dict[str, object] = {
        "status": "needs_review" if unchanged else "fail",
        "audit_version": AUDIT_VERSION,
        "v4_2_identity_exact": identity_exact,
        "reference_vs_oracle_direct_comparison": True,
        "target_mel_used": True,
        "target_f0_used": True,
        "target_voicing_used": True,
        "teacher_duration_grid_used": True,
        "predicted_duration_modified": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "training_started": False,
        "acoustic_training_authorized": False,
        "vocoder_training_authorized": False,
        "checkpoints_unchanged": unchanged,
        "items": items,
        "mean_v4_2_minus_reference": means,
        "interpretation": {
            "gangoso_dark": "centroid down + 300Hz-and-up deficit + low-band concentration",
            "robotic_buzzy": "voiced/unvoiced periodicity higher than reference",
            "mis_tuned": "generated voiced pitch MAE in cents increases versus reference",
            "lost_fricatives": "unvoiced 3k-8k energy deficit",
            "over_smoothed": "spectral flux lower than reference",
        },
        "next_gate": "review_reference_vs_v4_2_forensics_before_any_more_training",
        "report_path": str(report_path),
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v4_2_reference_waveform_forensics(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
