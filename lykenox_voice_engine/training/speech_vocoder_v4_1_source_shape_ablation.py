"""Full-utterance source-shape ablation for the LYKENOX v4.1 vocoder.

The first harmonic-gain ablation showed that simply reducing explicit harmonic authority is
not a viable fix: the 1.0 baseline is both the clearest and the loudest, while the buzz is
still present.  This diagnostic keeps the accepted v4.1 checkpoint frozen and isolates the
remaining source-shape hypotheses on one full held-out utterance with target mel/F0/voicing:

1. exact v4.1 baseline;
2. deterministic aperiodic source removed;
3. learned mel-conditioned harmonic envelope replaced by the neutral 1/h envelope;
4. only the first four harmonic channels retained, RMS-renormalized;
5. only the first two harmonic channels retained, RMS-renormalized.

A clean reference waveform from the same held-out recording is also written for A/B
listening.  Nothing in this module mutates a checkpoint or authorizes new training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import (
    _wave_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import (
    load_source_filter_checkpoint,
)


AUDIT_VERSION = "vocoder-v4-1-full-utterance-source-shape-ablation-v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _masked_harmonic_weights(
    weights: torch.Tensor,
    *,
    keep_harmonics: int | None,
) -> torch.Tensor:
    if keep_harmonics is None:
        return weights
    if keep_harmonics < 1 or keep_harmonics > int(weights.shape[1]):
        raise ValueError("keep_harmonics is outside the generator harmonic count")
    masked = weights.clone()
    masked[:, keep_harmonics:, :] = 0.0
    original_rms = torch.sqrt(weights.square().sum(dim=1, keepdim=True)).clamp_min(1e-8)
    masked_rms = torch.sqrt(masked.square().sum(dim=1, keepdim=True)).clamp_min(1e-8)
    return masked * (original_rms / masked_rms)


def _forward_source_variant(
    generator,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    *,
    noise_gain: float = 1.0,
    use_learned_harmonic_envelope: bool = True,
    keep_harmonics: int | None = None,
) -> torch.Tensor:
    """Reproduce v4.1 while changing only explicit source-shape components."""

    if noise_gain < 0.0:
        raise ValueError("noise_gain must be non-negative")
    generator._validate_inputs(mel, f0_hz, voiced)
    batch, mel_frames, _ = mel.shape
    samples = int(mel_frames) * generator.config.hop_length

    mel_samples = F.interpolate(
        mel.transpose(1, 2), size=samples, mode="linear", align_corners=False
    )
    f0_samples = F.interpolate(
        f0_hz.unsqueeze(1), size=samples, mode="linear", align_corners=False
    ).clamp_min(0.0)
    voiced_samples = F.interpolate(
        voiced.unsqueeze(1), size=samples, mode="linear", align_corners=False
    ).clamp(0.0, 1.0)

    if use_learned_harmonic_envelope:
        harmonic_weight_frames = generator._harmonic_weight_frames(mel)
    else:
        baseline = generator.baseline_harmonic_weights.view(1, generator.harmonics, 1).to(
            device=mel.device,
            dtype=mel.dtype,
        )
        harmonic_weight_frames = baseline.expand(batch, generator.harmonics, mel_frames)
    harmonic_weight_frames = _masked_harmonic_weights(
        harmonic_weight_frames,
        keep_harmonics=keep_harmonics,
    )
    harmonic_weights_samples = F.interpolate(
        harmonic_weight_frames, size=samples, mode="linear", align_corners=False
    )
    harmonic = generator._harmonic_source(
        f0_samples, voiced_samples, harmonic_weights_samples
    )

    noise = generator._aperiodic_source(
        batch,
        samples,
        device=mel.device,
        dtype=mel.dtype,
    )
    noise = noise * (0.08 + 0.92 * (1.0 - voiced_samples)) * float(noise_gain)
    log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)

    x = torch.cat(
        [mel_samples, harmonic, voiced_samples, log_f0, noise], dim=1
    )
    x = generator.input_projection(x)
    for block in generator.blocks:
        x = block(x)
    x = F.leaky_relu(x, negative_slope=0.1)
    raw_waveform = generator.post(x)
    filtered = F.conv1d(
        raw_waveform,
        generator.output_highpass_fir.to(
            device=raw_waveform.device,
            dtype=raw_waveform.dtype,
        ),
        padding=generator.highpass_kernel_size // 2,
    )
    waveform = torch.tanh(filtered).squeeze(1)
    if tuple(waveform.shape) != (batch, samples):
        raise RuntimeError("v4.1 source-shape ablation changed waveform length contract")
    return waveform


def _load_reference_waveform(
    wav_path: Path,
    *,
    sample_rate: int,
    samples: int,
) -> torch.Tensor:
    waveform, source_rate = load_audio(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(source_rate) != int(sample_rate):
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    wave = waveform[0].to(torch.float32)
    if int(wave.numel()) < samples:
        wave = F.pad(wave, (0, samples - int(wave.numel())))
    else:
        wave = wave[:samples]
    return wave.contiguous()


def run_v4_1_source_shape_ablation(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_source_filter_v4_1"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Accepted v4.1 checkpoint not found: {checkpoint}")

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v4_1_source_shape_ablation_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "source_shape_ablation_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator, _discriminator, payload = load_source_filter_checkpoint(checkpoint)
    generator.cpu().eval()

    speech_config = LykenoxSpeechConfig()
    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech_config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    item = dataset[0]
    base_item = dataset.base[0]
    batch = collate_aligned_speech([item]).to("cpu")
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("v4.1 source-shape ablation requires target F0/voicing")

    frames = int(batch.mel_lengths[0])
    samples = frames * generator.config.hop_length
    mel = batch.mel[:, :frames]
    f0_hz = batch.f0_hz[:, :frames]
    voiced = batch.voiced[:, :frames]

    reference = _load_reference_waveform(
        Path(str(base_item["wav_path"])),
        sample_rate=generator.config.sample_rate,
        samples=samples,
    )
    reference_path = output_dir / "00_reference.wav"
    sf.write(
        str(reference_path),
        reference.numpy(),
        generator.config.sample_rate,
        subtype="PCM_16",
    )

    variants = (
        ("01_baseline", dict()),
        ("02_noise_off", dict(noise_gain=0.0)),
        (
            "03_fixed_1_over_h_envelope",
            dict(use_learned_harmonic_envelope=False),
        ),
        ("04_keep_first_4_harmonics_renorm", dict(keep_harmonics=4)),
        ("05_keep_first_2_harmonics_renorm", dict(keep_harmonics=2)),
    )

    with torch.inference_mode():
        exact = generator(mel, f0_hz, voiced)
        reproduced = _forward_source_variant(generator, mel, f0_hz, voiced)
    reproduction_max_delta = float((exact - reproduced).abs().max())
    reproduction_exact = reproduction_max_delta <= 1e-7
    if not reproduction_exact:
        raise RuntimeError("Source-shape diagnostic does not reproduce v4.1 baseline exactly")

    variant_reports: list[dict[str, object]] = []
    with torch.inference_mode():
        for name, kwargs in variants:
            waveform = _forward_source_variant(
                generator,
                mel,
                f0_hz,
                voiced,
                **kwargs,
            )[0].detach().cpu().to(torch.float32).contiguous()
            if not bool(torch.isfinite(waveform).all()):
                raise RuntimeError(f"Non-finite waveform in source variant {name}")
            wav_path = output_dir / f"{name}.wav"
            sf.write(
                str(wav_path),
                waveform.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )
            variant_reports.append(
                {
                    "name": name,
                    "wav_path": str(wav_path),
                    **_wave_metrics(waveform, generator.config.sample_rate),
                }
            )

    report: dict[str, object] = {
        "status": "needs_listening",
        "audit_version": AUDIT_VERSION,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "generator_architecture": payload.get("generator_architecture"),
        "utterance_id": str(item["utterance_id"]),
        "text": str(item["text"]),
        "teacher_mel_frames": frames,
        "duration_seconds": round(samples / generator.config.sample_rate, 4),
        "target_voiced_fraction": round(float(voiced.mean()), 6),
        "baseline_reproduction_exact": reproduction_exact,
        "baseline_reproduction_max_delta": reproduction_max_delta,
        "reference": {
            "wav_path": str(reference_path),
            **_wave_metrics(reference, generator.config.sample_rate),
        },
        "variants": variant_reports,
        "interpretation": (
            "The previous gain sweep established that harmonic attenuation alone worsens "
            "clarity and loudness. Compare the exact baseline against noise_off to test the "
            "deterministic aperiodic source, against fixed_1_over_h_envelope to test learned "
            "harmonic spectral weighting, and against the 4/2-harmonic variants to test whether "
            "upper explicit harmonics are the dominant buzz source. The clean reference is the "
            "perceptual target. Do not retrain until this source-shape attribution is closed."
        ),
        "next_gate": "listen_source_shape_ablation_before_vocoder_revision",
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v4_1_source_shape_ablation(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
