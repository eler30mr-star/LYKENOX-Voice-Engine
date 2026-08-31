"""Read-only source/filter attribution for the colored v4.2 oracle renderer.

The accepted v4.2 checkpoint is driven by target mel, target F0 and target voicing on the
same three held-out utterances used by the direct reference-waveform forensics. Internal
source components are ablated without mutating model weights so the remaining waveform
error can be attributed to explicit harmonic excitation, deterministic aperiodic excitation,
or the mel-envelope/filter path. This audit does not train or normalize audio.
"""
from __future__ import annotations

import argparse
import json
import math
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
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import frame_grid_artifact_metrics
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import target_relative_presence_loss
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _load_reference_waveform,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_reference_waveform_forensics import (
    _atomic_json,
    _character_metrics,
    _pair_delta,
    _protected,
    _sha256,
)


AUDIT_VERSION = "vocoder-v4-2-source-filter-attribution-v1"
VALIDATION_INDICES = (0, 1, 2)
OUTPUT_DIR_NAME = "vocoder_v4_2_source_filter_attribution_v1"
VARIANTS = (
    "baseline",
    "no_harmonic",
    "no_aperiodic",
    "no_explicit_excitation",
    "mel_envelope_only",
)


def _forward_variant(vocoder, mel: torch.Tensor, f0_hz: torch.Tensor, voiced: torch.Tensor, variant: str):
    """Reproduce v4.2 exactly while optionally ablating source components in memory."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown v4.2 attribution variant: {variant}")
    vocoder._validate_inputs(mel, f0_hz, voiced)
    batch, mel_frames, _ = mel.shape
    samples = int(mel_frames) * vocoder.config.hop_length

    mel_frames_ch = mel.transpose(1, 2)
    conditioning_frames = vocoder.frame_conditioner(mel_frames_ch)
    conditioning_samples = F.interpolate(
        conditioning_frames,
        size=samples,
        mode="linear",
        align_corners=False,
    )
    envelope_hidden = vocoder.mel_to_hidden(conditioning_samples)

    f0_samples = F.interpolate(
        f0_hz.unsqueeze(1), size=samples, mode="linear", align_corners=False
    ).clamp_min(0.0)
    voiced_samples = F.interpolate(
        voiced.unsqueeze(1), size=samples, mode="linear", align_corners=False
    ).clamp(0.0, 1.0)
    harmonic_weights = F.interpolate(
        vocoder._harmonic_weight_frames(mel),
        size=samples,
        mode="linear",
        align_corners=False,
    )
    harmonic = vocoder._harmonic_source(f0_samples, voiced_samples, harmonic_weights)
    noise = vocoder._aperiodic_source(
        batch, samples, device=mel.device, dtype=mel.dtype
    )
    noise = noise * (0.08 + 0.92 * (1.0 - voiced_samples))

    if variant in ("no_harmonic", "no_explicit_excitation", "mel_envelope_only"):
        harmonic = torch.zeros_like(harmonic)
    if variant in ("no_aperiodic", "no_explicit_excitation", "mel_envelope_only"):
        noise = torch.zeros_like(noise)

    log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)
    source = torch.cat([harmonic, voiced_samples, log_f0, noise], dim=1)
    source_features = vocoder.source_stem(source)
    source_gate = torch.sigmoid(vocoder.source_gate(conditioning_samples))

    if variant == "mel_envelope_only":
        x = envelope_hidden
    else:
        x = envelope_hidden + source_gate * source_features

    skips: list[torch.Tensor] = []
    for block in vocoder.blocks:
        x, skip = block(x, conditioning_samples)
        skips.append(skip)
    x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(float(len(skips)))
    raw = vocoder.post(F.leaky_relu(x, negative_slope=0.1))
    filtered = F.conv1d(
        raw,
        vocoder.output_highpass_fir.to(device=raw.device, dtype=raw.dtype),
        padding=vocoder.highpass_kernel_size // 2,
    )
    waveform = torch.tanh(filtered).squeeze(1)
    if tuple(waveform.shape) != (batch, samples):
        raise RuntimeError("v4.2 attribution waveform length contract failed")
    diagnostics = {
        "source_gate_mean": float(source_gate.mean()),
        "source_gate_std": float(source_gate.std(unbiased=False)),
        "envelope_hidden_rms": float(torch.sqrt(envelope_hidden.square().mean())),
        "harmonic_source_rms": float(torch.sqrt(harmonic.square().mean())),
        "aperiodic_source_rms": float(torch.sqrt(noise.square().mean())),
        "source_features_rms": float(torch.sqrt(source_features.square().mean())),
    }
    return waveform, diagnostics


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _variant_summary(deltas: list[dict[str, float]]) -> dict[str, float]:
    if not deltas:
        return {}
    keys = list(deltas[0])
    result = {key: round(_mean([float(item[key]) for item in deltas]), 6) for key in keys}
    result["mean_abs_spectral_centroid_error_pct"] = round(
        _mean([abs(float(item["spectral_centroid_relative_pct"])) for item in deltas]), 6
    )
    result["mean_abs_voiced_pitch_delta_cents"] = round(
        _mean([abs(float(item["voiced_pitch_mae_cents_delta"])) for item in deltas]), 6
    )
    return result


def _improvement_counts(
    baseline: list[dict[str, float]],
    candidate: list[dict[str, float]],
) -> dict[str, int]:
    return {
        "spectral_envelope_300_4k_l1_db": sum(
            int(c["spectral_envelope_300_4k_l1_db"] < b["spectral_envelope_300_4k_l1_db"])
            for b, c in zip(baseline, candidate)
        ),
        "abs_spectral_centroid_error": sum(
            int(abs(c["spectral_centroid_relative_pct"]) < abs(b["spectral_centroid_relative_pct"]))
            for b, c in zip(baseline, candidate)
        ),
        "presence_1k_8k_error_db": sum(
            int(c["presence_1k_8k_error_db"] < b["presence_1k_8k_error_db"])
            for b, c in zip(baseline, candidate)
        ),
        "abs_voiced_pitch_delta": sum(
            int(abs(c["voiced_pitch_mae_cents_delta"]) < abs(b["voiced_pitch_mae_cents_delta"]))
            for b, c in zip(baseline, candidate)
        ),
    }


def run_v4_2_source_filter_attribution(root: Path) -> dict[str, object]:
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
        raise RuntimeError("source/filter attribution requires accepted v4.2")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root, "val", config, duration_root=duration_root, include_pitch_targets=True
    )
    if len(dataset) <= max(VALIDATION_INDICES):
        raise RuntimeError("not enough validation items for v4.2 attribution")

    output_dir = root / "models" / "lykenox_identity" / "evaluation" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "source_filter_attribution_report.json"
    variant_deltas: dict[str, list[dict[str, float]]] = {name: [] for name in VARIANTS}
    items: list[dict[str, object]] = []
    baseline_reproduction_exact = True
    structural_pass = True

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with torch.inference_mode():
        for audit_index, dataset_index in enumerate(VALIDATION_INDICES, start=1):
            item = dataset[dataset_index]
            raw_item = dataset.base[dataset_index]
            batch = collate_aligned_speech([item]).to("cpu")
            if batch.f0_hz is None or batch.voiced is None:
                raise RuntimeError("v4.2 attribution requires target F0/voicing")
            frames = int(batch.mel_lengths[0])
            samples = frames * config.hop_length
            if int(batch.durations[0].sum()) != frames:
                raise RuntimeError("teacher duration grid mismatch")
            target_mel = batch.mel[:, :frames]
            target_f0 = batch.f0_hz[:, :frames]
            target_voiced = batch.voiced[:, :frames]
            reference = _load_reference_waveform(
                Path(str(raw_item["wav_path"])), sample_rate=config.sample_rate, samples=samples
            )
            ref_metrics, ref_env, env_band = _character_metrics(
                reference,
                target_f0[0],
                target_voiced[0],
                sample_rate=config.sample_rate,
                hop=config.hop_length,
            )
            canonical = vocoder(target_mel, target_f0, target_voiced)
            reproduced, _ = _forward_variant(vocoder, target_mel, target_f0, target_voiced, "baseline")
            baseline_exact = torch.equal(canonical, reproduced)
            baseline_reproduction_exact = baseline_reproduction_exact and baseline_exact
            if not baseline_exact:
                raise RuntimeError("internal v4.2 baseline reproduction is not bit-exact")

            prefix = f"{audit_index:02d}"
            ref_path = output_dir / f"{prefix}_reference.wav"
            sf.write(str(ref_path), reference.numpy(), config.sample_rate, subtype="PCM_16")
            variant_payloads: dict[str, object] = {}
            for variant in VARIANTS:
                wave_batch, source_diag = _forward_variant(
                    vocoder, target_mel, target_f0, target_voiced, variant
                )
                valid = tuple(wave_batch.shape) == (1, samples) and bool(torch.isfinite(wave_batch).all())
                structural_pass = structural_pass and valid
                if not valid:
                    raise RuntimeError(f"invalid v4.2 attribution waveform: {variant}")
                wave = wave_batch[0].detach().cpu().to(torch.float32).contiguous()
                metrics, env, _ = _character_metrics(
                    wave,
                    target_f0[0],
                    target_voiced[0],
                    sample_rate=config.sample_rate,
                    hop=config.hop_length,
                )
                presence = target_relative_presence_loss(
                    wave.unsqueeze(0),
                    reference.unsqueeze(0),
                    sample_rate=config.sample_rate,
                    hop_length=config.hop_length,
                )
                grid = frame_grid_artifact_metrics(
                    wave,
                    sample_rate=config.sample_rate,
                    hop_length=config.hop_length,
                )
                delta = _pair_delta(ref_metrics, metrics, ref_env, env, env_band)
                delta["presence_1k_8k_error_db"] = abs(float(presence.presence_1k_8k_error_db))
                delta["frame_grid_harmonic_power_fraction"] = float(
                    grid.grid_harmonic_power_fraction[0]
                )
                delta["frame_grid_hop_autocorrelation"] = float(grid.hop_autocorrelation[0])
                delta["frame_grid_failure"] = float(bool(grid.severe_grid_artifact[0]))
                variant_deltas[variant].append(delta)
                wav_path = output_dir / f"{prefix}_{variant}.wav"
                sf.write(str(wav_path), wave.numpy(), config.sample_rate, subtype="PCM_16")
                variant_payloads[variant] = {
                    "wav_path": str(wav_path),
                    "source_diagnostics": {k: round(float(v), 6) for k, v in source_diag.items()},
                    "minus_reference": {k: round(float(v), 6) for k, v in delta.items()},
                }
            items.append({
                "audit_index": audit_index,
                "dataset_index": dataset_index,
                "utterance_id": str(item["utterance_id"]),
                "text": str(item["text"]),
                "reference_wav": str(ref_path),
                "baseline_reproduction_exact": baseline_exact,
                "variants": variant_payloads,
            })

    after = {name: _sha256(path) for name, path in protected.items()}
    checkpoints_unchanged = before == after
    means = {name: _variant_summary(variant_deltas[name]) for name in VARIANTS}
    baseline_mean = means["baseline"]
    relative: dict[str, object] = {}
    for name in VARIANTS[1:]:
        current = means[name]
        relative[name] = {
            "spectral_envelope_error_change_db": round(
                float(current["spectral_envelope_300_4k_l1_db"])
                - float(baseline_mean["spectral_envelope_300_4k_l1_db"]),
                6,
            ),
            "abs_centroid_error_change_pct": round(
                float(current["mean_abs_spectral_centroid_error_pct"])
                - float(baseline_mean["mean_abs_spectral_centroid_error_pct"]),
                6,
            ),
            "presence_error_change_db": round(
                float(current["presence_1k_8k_error_db"])
                - float(baseline_mean["presence_1k_8k_error_db"]),
                6,
            ),
            "abs_pitch_delta_change_cents": round(
                float(current["mean_abs_voiced_pitch_delta_cents"])
                - float(baseline_mean["mean_abs_voiced_pitch_delta_cents"]),
                6,
            ),
            "improvement_counts_vs_baseline": _improvement_counts(
                variant_deltas["baseline"], variant_deltas[name]
            ),
        }

    pass_gate = structural_pass and baseline_reproduction_exact and checkpoints_unchanged
    report: dict[str, object] = {
        "status": "pass" if pass_gate else "fail",
        "audit_version": AUDIT_VERSION,
        "v4_2_identity_exact": identity_exact,
        "baseline_reproduction_exact": baseline_reproduction_exact,
        "structural_pass": structural_pass,
        "checkpoints_unchanged": checkpoints_unchanged,
        "target_mel_used": True,
        "target_f0_used": True,
        "target_voicing_used": True,
        "teacher_duration_grid_used": True,
        "variants": list(VARIANTS),
        "mean_minus_reference_by_variant": means,
        "ablation_change_vs_baseline": relative,
        "items": items,
        "output_dir": str(output_dir),
        "training_started": False,
        "acoustic_training_authorized": False,
        "vocoder_training_authorized": False,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "decision_rule": (
            "Use the ablations only to localize which v4.2 subsystem causes the reference gap. "
            "Metrics may select a replacement architecture target but cannot accept voice quality."
        ),
        "next_gate": "select_vocoder_replacement_architecture_from_source_filter_attribution",
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v4_2_source_filter_attribution(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
