"""Full-utterance harmonic-source ablation for the LYKENOX v4.1 vocoder.

The reference-free integration audit exposed a persistent bee/buzz artifact.  The same
artifact is audible when the vocoder receives target mel + target F0/voicing + teacher
alignment durations, so this gate isolates the v4.1 source-filter itself before any new
training is authorized.

This diagnostic does not mutate the checkpoint or product runtime.  It reproduces the
v4.1 forward pass exactly, then scales only the explicit harmonic excitation and writes a
small listening set from one full held-out utterance.
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


AUDIT_VERSION = "vocoder-v4-1-full-utterance-harmonic-ablation-v1"
HARMONIC_GAINS = (1.0, 0.75, 0.50, 0.25, 0.0)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _forward_with_harmonic_gain(
    generator,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    *,
    harmonic_gain: float,
) -> torch.Tensor:
    """Reproduce v4.1 forward while scaling only explicit harmonic excitation."""

    if harmonic_gain < 0.0:
        raise ValueError("harmonic_gain must be non-negative")
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
    harmonic_weight_frames = generator._harmonic_weight_frames(mel)
    harmonic_weights_samples = F.interpolate(
        harmonic_weight_frames, size=samples, mode="linear", align_corners=False
    )

    harmonic = generator._harmonic_source(
        f0_samples, voiced_samples, harmonic_weights_samples
    ) * float(harmonic_gain)
    noise = generator._aperiodic_source(
        batch,
        samples,
        device=mel.device,
        dtype=mel.dtype,
    )
    noise = noise * (0.08 + 0.92 * (1.0 - voiced_samples))
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
            device=raw_waveform.device, dtype=raw_waveform.dtype
        ),
        padding=generator.highpass_kernel_size // 2,
    )
    waveform = torch.tanh(filtered).squeeze(1)
    if tuple(waveform.shape) != (batch, samples):
        raise RuntimeError("v4.1 harmonic ablation changed waveform length contract")
    return waveform


def run_v4_1_buzz_ablation(root: Path) -> dict[str, object]:
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
        / "vocoder_v4_1_buzz_ablation_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "buzz_ablation_report.json"

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator, _discriminator, payload = load_source_filter_checkpoint(checkpoint)
    generator.cpu().eval()

    # Use a complete held-out utterance, not the 64-frame crops used by the original
    # persistent v4.1 training/listening gate.
    from lykenox_voice_engine.models.speech import LykenoxSpeechConfig

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
    batch = collate_aligned_speech([item]).to("cpu")
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("v4.1 buzz ablation requires target F0/voicing")

    frames = int(batch.mel_lengths[0])
    mel = batch.mel[:, :frames]
    f0_hz = batch.f0_hz[:, :frames]
    voiced = batch.voiced[:, :frames]

    with torch.inference_mode():
        original = generator(mel, f0_hz, voiced)
        reproduced = _forward_with_harmonic_gain(
            generator, mel, f0_hz, voiced, harmonic_gain=1.0
        )
    reproduction_max_delta = float((original - reproduced).abs().max())
    reproduction_exact = reproduction_max_delta <= 1e-7
    if not reproduction_exact:
        raise RuntimeError(
            "Diagnostic v4.1 forward does not reproduce the checkpoint baseline exactly"
        )

    variants: list[dict[str, object]] = []
    with torch.inference_mode():
        for gain in HARMONIC_GAINS:
            waveform = _forward_with_harmonic_gain(
                generator,
                mel,
                f0_hz,
                voiced,
                harmonic_gain=gain,
            )[0].detach().cpu().to(torch.float32).contiguous()
            if not bool(torch.isfinite(waveform).all()):
                raise RuntimeError(f"Non-finite waveform at harmonic gain {gain}")
            gain_label = str(gain).replace(".", "p")
            wav_path = output_dir / f"harmonic_gain_{gain_label}.wav"
            sf.write(
                str(wav_path),
                waveform.numpy(),
                generator.config.sample_rate,
                subtype="PCM_16",
            )
            metrics = _wave_metrics(waveform, generator.config.sample_rate)
            bands = metrics.get("band_power_fraction", {})
            above_300 = 0.0
            if isinstance(bands, dict):
                above_300 = float(bands.get("300_3000", 0.0)) + float(
                    bands.get("3000_nyquist", 0.0)
                )
            variants.append(
                {
                    "harmonic_gain": gain,
                    "wav_path": str(wav_path),
                    **metrics,
                    "above_300hz_fraction": round(above_300, 6),
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
        "duration_seconds": round(
            frames * generator.config.hop_length / generator.config.sample_rate, 4
        ),
        "target_voiced_fraction": round(float(voiced.mean()), 6),
        "baseline_reproduction_exact": reproduction_exact,
        "baseline_reproduction_max_delta": reproduction_max_delta,
        "training_segment_mel_frames": 64,
        "full_utterance_to_training_segment_ratio": round(frames / 64.0, 4),
        "variants": variants,
        "interpretation": (
            "If the bee/buzz artifact weakens materially as harmonic_gain decreases while "
            "speech remains recognizable, explicit harmonic excitation is the primary v4.1 "
            "failure source. This is a diagnostic ablation only; do not ship a gain tweak or "
            "retrain until the listening result is assigned."
        ),
        "next_gate": "listen_harmonic_gain_ablation_before_vocoder_change",
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v4_1_buzz_ablation(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
