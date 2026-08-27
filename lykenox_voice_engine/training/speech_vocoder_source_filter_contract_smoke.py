"""Bounded persistent-checkpoint smoke for the LYKENOX v4.1 vocoder candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderGeneratorV41,
    VOCODER_GENERATOR_V4_1_ARCHITECTURE,
)
from lykenox_voice_engine.training.speech_pitch import extract_pitch_frames
from lykenox_voice_engine.training.speech_vocoder_data import collect_vocoder_segments
from lykenox_voice_engine.training.speech_vocoder_losses import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)
from lykenox_voice_engine.training.speech_vocoder_source_balance import target_relative_spectral_balance_loss
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import (
    build_source_filter_training_provenance,
    load_source_filter_checkpoint,
    save_source_filter_checkpoint,
)


SMOKE_VERSION = "source-filter-persistent-contract-smoke-v1"
DEFAULT_TIME_BUDGET_SECONDS = 85.0


def _condition(segment):
    pitch = extract_pitch_frames(segment.waveform, frame_count=segment.mel_frames)
    return pitch.f0_hz.unsqueeze(0), pitch.voiced.unsqueeze(0)


def run_smoke(root: Path, *, steps: int = 16, time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS) -> dict[str, object]:
    root = Path(root).resolve()
    started = time.perf_counter()
    torch.manual_seed(1337)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    segments, _ = collect_vocoder_segments(root, "train", segment_mel_frames=64, max_items=4, seed=1337)
    generator = LykenoxVocoderGeneratorV41().cpu().train()
    discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2).cpu().train()
    g_opt = torch.optim.AdamW(generator.parameters(), lr=2e-4, weight_decay=1e-5)
    d_opt = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, weight_decay=1e-5)

    probe = segments[0]
    probe_f0, probe_voiced = _condition(probe)
    with torch.no_grad():
        before_wave = generator(probe.mel.unsqueeze(0), probe_f0, probe_voiced)
        before_recon = float(multi_resolution_reconstruction_loss(before_wave, probe.waveform.unsqueeze(0)).total)
        before_balance = float(target_relative_spectral_balance_loss(before_wave, probe.waveform.unsqueeze(0)).loss)

    update_times: list[float] = []
    last_recon = math.inf
    for step in range(steps):
        if time.perf_counter() - started >= time_budget_seconds:
            raise RuntimeError("v4.1 checkpoint smoke exceeded bounded time budget")
        segment = segments[step % len(segments)]
        f0, voiced = _condition(segment)
        mel = segment.mel.unsqueeze(0)
        target = segment.waveform.unsqueeze(0)
        update_started = time.perf_counter()

        if step >= steps // 2:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            d_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                detached = generator(mel, f0, voiced)
            d_loss = discriminator_hinge_loss(discriminator(target), discriminator(detached))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
            d_opt.step()

        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        g_opt.zero_grad(set_to_none=True)
        prediction = generator(mel, f0, voiced)
        reconstruction = multi_resolution_reconstruction_loss(prediction, target)
        balance = target_relative_spectral_balance_loss(prediction, target)
        loss = reconstruction.total + 0.50 * balance.loss
        if step >= steps // 2:
            with torch.no_grad():
                real_features = discriminator(target)
            fake_features = discriminator(prediction)
            loss = loss + 0.05 * generator_adversarial_loss(fake_features) + 1.0 * feature_matching_loss(real_features, fake_features)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite v4.1 persistent smoke loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        if not math.isfinite(float(grad)):
            raise RuntimeError("Non-finite v4.1 persistent smoke gradient")
        g_opt.step()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
        last_recon = float(reconstruction.total.detach())
        update_times.append(time.perf_counter() - update_started)

    with torch.no_grad():
        after_wave = generator(probe.mel.unsqueeze(0), probe_f0, probe_voiced)
        after_recon = float(multi_resolution_reconstruction_loss(after_wave, probe.waveform.unsqueeze(0)).total)
        after_balance = float(target_relative_spectral_balance_loss(after_wave, probe.waveform.unsqueeze(0)).loss)

    provenance = build_source_filter_training_provenance(root, segment_mel_frames=64, seed=1337)
    artifact_dir = root / "models" / "lykenox_identity" / "training" / "source_filter_contract_smoke"
    checkpoint_path = artifact_dir / "roundtrip.pt"
    save_source_filter_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        epoch=3,
        global_step=steps,
        next_item_offset=2,
        validation_reconstruction_loss=after_recon,
        validation_spectral_balance_loss=after_balance,
        validation_selection_score=after_recon + 0.50 * after_balance,
        training_provenance=provenance,
        generator_optimizer=g_opt,
        discriminator_optimizer=d_opt,
        training_metadata={"smoke_version": SMOKE_VERSION},
    )
    loaded_generator, _, payload = load_source_filter_checkpoint(checkpoint_path)
    with torch.no_grad():
        roundtrip_wave = loaded_generator(probe.mel.unsqueeze(0), probe_f0, probe_voiced)
    max_delta = float((after_wave - roundtrip_wave).abs().max())

    optimizer_states_present = bool(payload.get("generator_optimizer_state")) and bool(payload.get("discriminator_optimizer_state"))
    resume_metadata_exact = payload.get("epoch") == 3 and payload.get("global_step") == steps and payload.get("next_item_offset") == 2
    provenance_exact = payload.get("training_provenance") == provenance
    architecture_exact = payload.get("generator_architecture") == VOCODER_GENERATOR_V4_1_ARCHITECTURE
    checkpoint_roundtrip_exact = max_delta == 0.0
    status = "pass" if optimizer_states_present and resume_metadata_exact and provenance_exact and architecture_exact and checkpoint_roundtrip_exact else "needs_review"

    report = {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "architecture": VOCODER_GENERATOR_V4_1_ARCHITECTURE,
        "parameters": generator.parameter_count(),
        "steps": steps,
        "probe_reconstruction_before": round(before_recon, 6),
        "probe_reconstruction_after": round(after_recon, 6),
        "probe_spectral_balance_before": round(before_balance, 6),
        "probe_spectral_balance_after": round(after_balance, 6),
        "last_training_reconstruction": round(last_recon, 6),
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "checkpoint_waveform_max_abs_delta": max_delta,
        "checkpoint_provenance_exact": provenance_exact,
        "checkpoint_architecture_exact": architecture_exact,
        "optimizer_states_present": optimizer_states_present,
        "resume_metadata_exact": resume_metadata_exact,
        "mean_seconds_per_update": round(statistics.fmean(update_times), 4),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint_path),
        "next_gate": "build_bounded_resumable_v4_1_trainer" if status == "pass" else "fix_v4_1_checkpoint_contract",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "contract_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.root, steps=args.steps, time_budget_seconds=args.time_budget_seconds), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
