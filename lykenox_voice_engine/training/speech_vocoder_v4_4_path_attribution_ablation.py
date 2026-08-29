"""Post-training path attribution for perceptually rejected LYKENOX vocoder v4.4.

V4.4 improved target-relative harmonic-exposure and broad spectral-balance metrics, but
full-utterance listening did not improve the radio-mistuned / metallic interference.
Before designing or training another vocoder, this bounded audit isolates which trained
v4.4 path carries the remaining artifact.

The audit uses one complete held-out oracle-conditioned utterance and never trains or
mutates checkpoints.  It compares:

* exact trained v4.4 baseline;
* periodic excitation only;
* aperiodic excitation only;
* both excitations but no per-block aperiodic residual injection;
* both excitations with the dynamic filter selector forced to an equal basis mixture.

A custom diagnostic forward must first reproduce the normal generator sample-for-sample.
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
from lykenox_voice_engine.models.vocoder import VOCODER_GENERATOR_V4_4_ARCHITECTURE
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)
from lykenox_voice_engine.training.speech_reference_free_failure_isolation_audit import (
    _wave_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_v4_4_artifact import load_v4_4_checkpoint


AUDIT_VERSION = "vocoder-v4-4-path-attribution-ablation-v1"
VALIDATION_CANDIDATES = (0, 1, 2)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dynamic_block_forward(
    block,
    x: torch.Tensor,
    aperiodic_state: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    equal_filter_selector: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    activated = F.leaky_relu(x, negative_slope=0.1)
    candidates = torch.stack([layer(activated) for layer in block.filters], dim=2)
    batch, channels, _bases, samples = candidates.shape
    if equal_filter_selector:
        selector = torch.full(
            (batch, channels, block.filter_bases, samples),
            1.0 / float(block.filter_bases),
            device=x.device,
            dtype=x.dtype,
        )
    else:
        selector = block.filter_selector(conditioning).view(
            batch,
            channels,
            block.filter_bases,
            samples,
        )
        selector = torch.softmax(selector, dim=2)
    dynamic = (candidates * selector).sum(dim=2)
    dynamic = block.channel_mix(F.leaky_relu(dynamic, negative_slope=0.1))
    noise_gate = torch.sigmoid(block.aperiodic_gate(conditioning))
    noise_detail = block.aperiodic_projection(aperiodic_state) * noise_gate
    y = torch.tanh(dynamic + noise_detail)
    residual = (x + block.residual_projection(y)) * (2.0 ** -0.5)
    skip = block.skip_projection(y)
    return residual, skip


def _diagnostic_forward(
    generator,
    mel: torch.Tensor,
    f0_hz: torch.Tensor,
    voiced: torch.Tensor,
    *,
    periodic_enabled: bool = True,
    aperiodic_enabled: bool = True,
    block_aperiodic_enabled: bool = True,
    equal_filter_selector: bool = False,
) -> torch.Tensor:
    generator._validate_inputs(mel, f0_hz, voiced)
    batch, mel_frames, _ = mel.shape
    samples = int(mel_frames) * generator.config.hop_length

    conditioning_frames = generator.frame_conditioner(mel.transpose(1, 2))
    conditioning = F.interpolate(
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

    harmonic = generator._harmonic_carrier(f0_samples, voiced_samples)
    log_f0 = torch.log1p(f0_samples) / math.log1p(500.0)
    periodic_input = torch.cat([harmonic, voiced_samples, log_f0], dim=1)

    noise = generator._aperiodic_source(
        batch,
        samples,
        device=mel.device,
        dtype=mel.dtype,
    )
    voiced_noise_floor = 0.12
    shaped_noise = noise * (
        voiced_noise_floor + (1.0 - voiced_noise_floor) * (1.0 - voiced_samples)
    )
    aperiodic_input = torch.cat([shaped_noise, 1.0 - voiced_samples], dim=1)

    periodic_state = generator.periodic_stem(periodic_input)
    aperiodic_state = generator.aperiodic_stem(aperiodic_input)
    if not periodic_enabled:
        periodic_state = torch.zeros_like(periodic_state)
    if not aperiodic_enabled:
        aperiodic_state = torch.zeros_like(aperiodic_state)

    x = generator.initial_mix(torch.cat([periodic_state, aperiodic_state], dim=1))
    block_aperiodic = (
        aperiodic_state if block_aperiodic_enabled else torch.zeros_like(aperiodic_state)
    )
    skips: list[torch.Tensor] = []
    for block in generator.blocks:
        x, skip = _dynamic_block_forward(
            block,
            x,
            block_aperiodic,
            conditioning,
            equal_filter_selector=equal_filter_selector,
        )
        skips.append(skip)
    x = torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(float(len(skips)))
    raw = generator.post(x)
    filtered = F.conv1d(
        raw,
        generator.output_highpass_fir.to(device=raw.device, dtype=raw.dtype),
        padding=generator.highpass_kernel_size // 2,
    )
    waveform = torch.tanh(filtered).squeeze(1)
    if tuple(waveform.shape) != (batch, samples):
        raise RuntimeError("v4.4 diagnostic output length mismatch")
    return waveform


def run_v4_4_path_attribution_ablation(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_dynamic_filter_hybrid_v4_4"
        / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Persistent v4.4 best checkpoint not found: {checkpoint}")
    checkpoint_hash_before = _sha256(checkpoint)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    generator, _discriminator, payload = load_v4_4_checkpoint(checkpoint)
    if (
        generator.architecture != VOCODER_GENERATOR_V4_4_ARCHITECTURE
        or payload.get("generator_architecture") != VOCODER_GENERATOR_V4_4_ARCHITECTURE
    ):
        raise RuntimeError("Path attribution requires the trained v4.4 architecture")
    generator.cpu().eval()

    speech = LykenoxSpeechConfig()
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "val",
        speech,
        duration_root=find_clean_duration_root(root),
        include_pitch_targets=True,
    )
    candidates: list[tuple[int, object, int]] = []
    for dataset_index in VALIDATION_CANDIDATES:
        item = dataset[dataset_index]
        batch = collate_aligned_speech([item]).to("cpu")
        candidates.append((dataset_index, item, int(batch.mel_lengths[0])))
    dataset_index, item, _candidate_frames = min(candidates, key=lambda row: row[2])
    batch = collate_aligned_speech([item]).to("cpu")
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Path attribution requires cached target F0/voicing")
    frames = int(batch.mel_lengths[0])
    mel = batch.mel[:, :frames]
    f0_hz = batch.f0_hz[:, :frames]
    voiced = batch.voiced[:, :frames]
    expected_samples = frames * speech.hop_length

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "vocoder_v4_4_path_attribution_ablation_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "path_attribution_report.json"

    variants = (
        ("baseline", {}),
        ("periodic_only", {"aperiodic_enabled": False}),
        ("aperiodic_only", {"periodic_enabled": False}),
        ("no_block_aperiodic", {"block_aperiodic_enabled": False}),
        ("equal_filter_selector", {"equal_filter_selector": True}),
    )

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        normal = generator(mel, f0_hz, voiced)
        diagnostic_baseline = _diagnostic_forward(generator, mel, f0_hz, voiced)
        max_delta = float((normal - diagnostic_baseline).abs().max())
        baseline_exact = max_delta == 0.0
        if not baseline_exact:
            raise RuntimeError(
                f"v4.4 path diagnostic failed exact baseline reproduction: {max_delta}"
            )
        for name, kwargs in variants:
            waveform = (
                diagnostic_baseline
                if name == "baseline"
                else _diagnostic_forward(generator, mel, f0_hz, voiced, **kwargs)
            )
            if tuple(waveform.shape) != (1, expected_samples):
                raise RuntimeError(f"v4.4 path variant length mismatch: {name}")
            if not bool(torch.isfinite(waveform).all()):
                raise RuntimeError(f"v4.4 path variant non-finite: {name}")
            wave = waveform[0].detach().cpu().to(torch.float32).contiguous()
            wav_path = output_dir / f"{name}.wav"
            sf.write(str(wav_path), wave.numpy(), speech.sample_rate, subtype="PCM_16")
            rows.append({
                "name": name,
                "wav_path": str(wav_path),
                **_wave_metrics(wave, speech.sample_rate),
            })

    checkpoint_hash_after = _sha256(checkpoint)
    checkpoint_unchanged = checkpoint_hash_before == checkpoint_hash_after
    report: dict[str, object] = {
        "status": "needs_listening" if checkpoint_unchanged else "fail",
        "audit_version": AUDIT_VERSION,
        "architecture": VOCODER_GENERATOR_V4_4_ARCHITECTURE,
        "dataset_index": dataset_index,
        "utterance_id": str(item["utterance_id"]),
        "mel_frames": frames,
        "duration_seconds": round(expected_samples / speech.sample_rate, 4),
        "baseline_reproduction_exact": baseline_exact,
        "baseline_reproduction_max_delta": max_delta,
        "checkpoint_unchanged": checkpoint_unchanged,
        "persistent_training_restarted": False,
        "oracle_conditioning_used_for_audit_only": True,
        "reference_audio_required_for_product_inference": False,
        "variants": rows,
        "listening_order": [name for name, _kwargs in variants],
        "decision_rule": (
            "If periodic_only preserves the radio-like artifact while aperiodic_only does not, "
            "the periodic path remains dominant. If aperiodic_only preserves it, the aperiodic "
            "path/filtering is implicated. If equal_filter_selector materially cleans the voice, "
            "time-varying filter selection is generating the artifact. If no_block_aperiodic "
            "cleans it, repeated per-block aperiodic injection is implicated. Do not train a new "
            "vocoder until one of these path-level attributions is established."
        ),
        "next_gate": "listen_v4_4_path_attribution_before_any_new_vocoder_training",
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_v4_4_path_attribution_ablation(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
