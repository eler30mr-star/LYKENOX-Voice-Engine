"""Full-utterance carrier fine-structure ablation for trained LYKENOX vocoder v4.3.

Listening rejected the persistent v4.3 oracle output even though its held-out mel-envelope
metrics improved.  The remaining question is whether the regression is caused mainly by the
new 24-harmonic deterministic carrier / low voiced-noise ratio, or by the stricter
multiplicative-only mel filter itself.

This audit does not train or mutate checkpoints.  It uses one complete held-out utterance
with target mel + target F0 + target voicing and changes only carrier fine structure before
the already-trained v4.3 filter:

* exact trained 24-harmonic / 0.05 voiced-noise baseline;
* 16, 12, and 8 active harmonics with active weights re-normalized to preserve harmonic RMS;
* 24 harmonics with voiced-noise floors 0.10 and 0.20.

The exact baseline is reproduced through the diagnostic forward and compared sample-for-
sample with the model's normal forward.  No post-hoc waveform normalization is applied.
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
from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V4_3_ARCHITECTURE
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import _wave_metrics
from lykenox_voice_engine.training.speech_vocoder_local_spectral_contrast import (
    target_relative_local_spectral_contrast_loss,
)
from lykenox_voice_engine.training.speech_vocoder_v4_3_artifact import load_v4_3_checkpoint


AUDIT_VERSION = "vocoder-v4-3-carrier-fine-structure-ablation-v1"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _custom_harmonic_carrier(
    generator,
    f0_samples: torch.Tensor,
    voiced_samples: torch.Tensor,
    *,
    active_harmonics: int,
) -> torch.Tensor:
    if not 1 <= active_harmonics <= generator.harmonics:
        raise ValueError("active_harmonics outside trained carrier range")
    phase = torch.cumsum(
        2.0 * math.pi * f0_samples / float(generator.config.sample_rate),
        dim=2,
    )
    weights = generator.harmonic_weights.to(
        device=f0_samples.device,
        dtype=f0_samples.dtype,
    ).clone()
    active = weights[:active_harmonics]
    active = active / torch.sqrt(active.square().sum()).clamp_min(1e-8)
    weights.zero_()
    weights[:active_harmonics] = active

    guard_hz = 0.46 * float(generator.config.sample_rate)
    transition_hz = 350.0
    channels: list[torch.Tensor] = []
    for harmonic_index in range(1, generator.harmonics + 1):
        if harmonic_index > active_harmonics:
            channels.append(torch.zeros_like(f0_samples))
            continue
        offset = (harmonic_index * 0.61803398875 % 1.0) * 2.0 * math.pi
        frequency = f0_samples * float(harmonic_index)
        anti_alias = torch.sigmoid((guard_hz - frequency) / transition_hz)
        source = torch.sin(phase * harmonic_index + offset)
        channels.append(
            source
            * voiced_samples
            * anti_alias
            * weights[harmonic_index - 1]
        )
    return torch.cat(channels, dim=1)


def _diagnostic_forward(
    generator,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    *,
    active_harmonics: int,
    voiced_noise_floor: float,
    exact_trained_harmonics: bool,
) -> torch.Tensor:
    generator._validate_inputs(mel, f0_hz, voiced)
    batch, mel_frames, _ = mel.shape
    samples = int(mel_frames) * generator.config.hop_length

    conditioning_frames = generator.frame_conditioner(mel.transpose(1, 2))
    conditioning_samples = F.interpolate(
        conditioning_frames,
        size=samples,
        mode="linear",
        align_corners=False,
    )
    f0_samples = F.interpolate(
        f0_hz.unsqueeze(1),
        size=samples,
        mode="linear",
        align_corners=False,
    ).clamp_min(0.0)
    voiced_samples = F.interpolate(
        voiced.unsqueeze(1),
        size=samples,
        mode="linear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    if exact_trained_harmonics:
        harmonic = generator._carrier(f0_samples, voiced_samples)
    else:
        harmonic = _custom_harmonic_carrier(
            generator,
            f0_samples,
            voiced_samples,
            active_harmonics=active_harmonics,
        )

    noise = generator._aperiodic_source(
        batch,
        samples,
        device=mel.device,
        dtype=mel.dtype,
    )
    noise = noise * (
        float(voiced_noise_floor)
        + (1.0 - float(voiced_noise_floor)) * (1.0 - voiced_samples)
    )
    log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)
    carrier = torch.cat([harmonic, noise, voiced_samples, log_f0], dim=1)

    x = generator.carrier_stem(carrier)
    skips: list[torch.Tensor] = []
    for block in generator.blocks:
        x, skip = block(x, conditioning_samples)
        skips.append(skip)
    x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(float(len(skips)))
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
        raise RuntimeError("v4.3 diagnostic output length mismatch")
    return waveform


def run_v4_3_carrier_fine_structure_ablation(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_mel_filtered_carrier_v4_3"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Trained v4.3 best checkpoint not found: {checkpoint}")
    checkpoint_hash_before = _sha256(checkpoint)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator, _discriminator, payload = load_v4_3_checkpoint(checkpoint)
    if (
        generator.architecture != VOCODER_GENERATOR_V4_3_ARCHITECTURE
        or payload.get("generator_architecture") != VOCODER_GENERATOR_V4_3_ARCHITECTURE
    ):
        raise RuntimeError("Carrier ablation requires the trained v4.3 architecture")
    generator.cpu().eval()

    speech = LykenoxSpeechConfig()
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech,
        duration_root=find_clean_duration_root(root),
        include_pitch_targets=True,
    )
    item = dataset[0]
    batch = collate_aligned_speech([item]).to("cpu")
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Carrier ablation requires target pitch cache")
    frames = int(batch.mel_lengths[0])
    mel = batch.mel[:, :frames]
    f0_hz = batch.f0_hz[:, :frames]
    voiced = batch.voiced[:, :frames]

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v4_3_carrier_fine_structure_ablation_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "carrier_fine_structure_ablation_report.json"

    variants = (
        ("baseline_24h_noise005", 24, 0.05, True),
        ("harmonics_16_equal_rms", 16, 0.05, False),
        ("harmonics_12_equal_rms", 12, 0.05, False),
        ("harmonics_8_equal_rms", 8, 0.05, False),
        ("noise_floor_0p10", 24, 0.10, True),
        ("noise_floor_0p20", 24, 0.20, True),
    )

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        trained = generator(mel, f0_hz, voiced)
        baseline = _diagnostic_forward(
            generator,
            mel,
            f0_hz,
            voiced,
            active_harmonics=24,
            voiced_noise_floor=0.05,
            exact_trained_harmonics=True,
        )
        max_delta = float((trained - baseline).abs().max())
        baseline_exact = max_delta == 0.0
        if not baseline_exact:
            raise RuntimeError(
                f"v4.3 carrier diagnostic failed exact baseline reproduction: {max_delta}"
            )

        for name, active_harmonics, noise_floor, exact_harmonics in variants:
            wave_batch = _diagnostic_forward(
                generator,
                mel,
                f0_hz,
                voiced,
                active_harmonics=active_harmonics,
                voiced_noise_floor=noise_floor,
                exact_trained_harmonics=exact_harmonics,
            )
            wave = wave_batch[0].detach().cpu().to(torch.float32).contiguous()
            if not bool(torch.isfinite(wave).all()):
                raise RuntimeError(f"Non-finite carrier ablation waveform: {name}")
            wav_path = output_dir / f"{name}.wav"
            sf.write(str(wav_path), wave.numpy(), speech.sample_rate, subtype="PCM_16")
            contrast = target_relative_local_spectral_contrast_loss(
                wave_batch,
                batch.mel.new_zeros(wave_batch.shape),
                hop_length=speech.hop_length,
            )
            # The contrast-to-zero number is only a compact fine-structure descriptor here;
            # listening is the decision gate, and no variant is accepted numerically.
            rows.append(
                {
                    "name": name,
                    "active_harmonics": active_harmonics,
                    "voiced_noise_floor": noise_floor,
                    "wav_path": str(wav_path),
                    **_wave_metrics(wave, speech.sample_rate),
                    "mean_abs_local_contrast_descriptor": round(
                        float(contrast.prediction_mean_abs_contrast), 6
                    ),
                }
            )

    checkpoint_hash_after = _sha256(checkpoint)
    report: dict[str, object] = {
        "status": "needs_listening",
        "audit_version": AUDIT_VERSION,
        "generator_architecture": VOCODER_GENERATOR_V4_3_ARCHITECTURE,
        "checkpoint": str(checkpoint),
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "baseline_reproduction_exact": baseline_exact,
        "baseline_reproduction_max_delta": max_delta,
        "persistent_training_restarted": False,
        "utterance_id": str(item["utterance_id"]),
        "text": str(item["text"]),
        "teacher_mel_frames": frames,
        "duration_seconds": round(frames * speech.hop_length / speech.sample_rate, 4),
        "variants": rows,
        "interpretation": (
            "Listen first to baseline_24h_noise005, then harmonic truncations, then the two "
            "higher voiced-noise variants. If equal-RMS harmonic truncation materially removes "
            "the regression while speech remains useful, the 24-harmonic carrier is too rich/" 
            "exposed for this filter. If higher voiced noise reduces the metallic periodicity "
            "without destroying intelligibility, excessive phase-coherent excitation is the "
            "primary issue. If neither family helps, the multiplicative-only v4.3 filter is too "
            "restrictive to transform deterministic carrier fine structure into natural speech, "
            "and the next architecture must add controlled learned nonperiodic/residual capacity "
            "without restoring an additive raw-carrier shortcut."
        ),
        "next_gate": "listen_v4_3_carrier_fine_structure_ablation_before_any_v4_4_training",
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_v4_3_carrier_fine_structure_ablation(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
