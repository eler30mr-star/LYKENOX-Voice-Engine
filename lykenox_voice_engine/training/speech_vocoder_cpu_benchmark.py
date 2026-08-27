"""CPU feasibility gate for the first LYKENOX-owned neural vocoder.

This is deliberately a bounded benchmark, not a production vocoder training run. It uses
real cached LYKENOX mels and their original local waveform, verifies exact hop-length
upsampling, exercises generator backprop with a small waveform + spectral objective, and
measures inference real-time factor on the target CPU.

Passing proves that an owned mel -> waveform neural path is mechanically trainable and can
run at useful local speed. It does not prove final perceptual quality; adversarial/multi-
resolution objectives and a persistent vocoder checkpoint remain later gates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import torchaudio

from lykenox_voice_engine.audio.io import load_audio
from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.models.vocoder import LykenoxVocoderConfig, LykenoxVocoderGenerator
from lykenox_voice_engine.training.speech_aligner_train import _dataset


def _mono_24k_waveform(path: Path, config: LykenoxVocoderConfig) -> torch.Tensor:
    waveform, sample_rate = load_audio(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            config.sample_rate,
        )
    peak = waveform.abs().max().clamp_min(1e-8)
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.squeeze(0).to(torch.float32).contiguous()


def _wave_segment(
    waveform: torch.Tensor,
    *,
    start_frame: int,
    mel_frames: int,
    hop_length: int,
) -> torch.Tensor:
    start = start_frame * hop_length
    samples = mel_frames * hop_length
    segment = waveform[start : start + samples]
    if segment.numel() < samples:
        segment = F.pad(segment, (0, samples - int(segment.numel())))
    return segment


def _spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Small differentiable benchmark objective; not the final vocoder recipe."""

    waveform_loss = F.l1_loss(prediction, target)
    spectral_terms: list[torch.Tensor] = []
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
        pred_stft = torch.stft(
            prediction,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        pred_mag = pred_stft.abs().clamp_min(1e-5)
        target_mag = target_stft.abs().clamp_min(1e-5)
        spectral_terms.append(F.l1_loss(torch.log(pred_mag), torch.log(target_mag)))
    return waveform_loss + 0.25 * torch.stack(spectral_terms).mean()


def _inference_benchmark(
    model: LykenoxVocoderGenerator,
    mel: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    repeats: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _ = model(mel.unsqueeze(0))
        timings: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            waveform = model(mel.unsqueeze(0))
            timings.append(time.perf_counter() - started)
    audio_seconds = int(waveform.shape[1]) / sample_rate
    median_seconds = statistics.median(timings)
    return {
        "audio_seconds": round(audio_seconds, 4),
        "median_inference_seconds": round(median_seconds, 4),
        "mean_inference_seconds": round(statistics.fmean(timings), 4),
        "min_inference_seconds": round(min(timings), 4),
        "max_inference_seconds": round(max(timings), 4),
        "median_realtime_factor": round(median_seconds / max(audio_seconds, 1e-9), 4),
        "median_realtime_multiple": round(audio_seconds / max(median_seconds, 1e-9), 2),
        "waveform_samples": int(waveform.shape[1]),
        "samples_per_mel_frame": hop_length,
    }


def run_vocoder_cpu_benchmark(
    root: Path,
    *,
    train_steps: int = 8,
    train_mel_frames: int = 32,
    inference_mel_frames: int = 96,
    inference_repeats: int = 5,
) -> dict[str, object]:
    if train_steps < 1 or train_mel_frames < 8 or inference_mel_frames < 16:
        raise ValueError("Benchmark bounds are unrealistically small")
    if inference_repeats < 2:
        raise ValueError("inference_repeats must be >= 2")

    root = Path(root).resolve()
    torch.manual_seed(1337)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    speech_config = LykenoxSpeechConfig()
    vocoder_config = LykenoxVocoderConfig(
        mel_bins=speech_config.mel_bins,
        sample_rate=speech_config.sample_rate,
        hop_length=speech_config.hop_length,
    )
    dataset = _dataset(root, "train", speech_config)
    if len(dataset) == 0:
        raise RuntimeError("Speech dataset is empty")

    selected = None
    minimum_frames = max(train_mel_frames, inference_mel_frames) + 8
    for index in range(len(dataset)):
        item = dataset[index]
        if int(item["mel"].shape[0]) >= minimum_frames:
            selected = item
            break
    if selected is None:
        raise RuntimeError(
            f"No real utterance contains at least {minimum_frames} mel frames"
        )

    full_mel = selected["mel"].to(torch.float32)
    wav_path = Path(str(selected["wav_path"]))
    waveform = _mono_24k_waveform(wav_path, vocoder_config)

    # Avoid the centered-STFT recording boundary during the bounded optimization probe.
    start_frame = 4
    train_mel = full_mel[start_frame : start_frame + train_mel_frames]
    target_wave = _wave_segment(
        waveform,
        start_frame=start_frame,
        mel_frames=train_mel_frames,
        hop_length=vocoder_config.hop_length,
    )
    inference_mel = full_mel[
        start_frame : start_frame + inference_mel_frames
    ]

    model = LykenoxVocoderGenerator(vocoder_config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    losses: list[float] = []
    timings: list[float] = []
    max_grad_norm = 0.0
    exact_length = True
    expected_train_samples = train_mel_frames * vocoder_config.hop_length

    for step in range(train_steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        predicted = model(train_mel.unsqueeze(0)).squeeze(0)
        exact_length = exact_length and int(predicted.numel()) == expected_train_samples
        loss = _spectral_loss(predicted, target_wave)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite vocoder benchmark loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"Non-finite vocoder gradient norm at step {step}")
        optimizer.step()
        timings.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
        max_grad_norm = max(max_grad_norm, float(grad_norm))

    inference = _inference_benchmark(
        model,
        inference_mel,
        sample_rate=vocoder_config.sample_rate,
        hop_length=vocoder_config.hop_length,
        repeats=inference_repeats,
    )
    loss_decreased = losses[-1] < losses[0]
    realtime = float(inference["median_realtime_factor"]) <= 1.0
    gate_pass = exact_length and loss_decreased and realtime

    report = {
        "status": "pass" if gate_pass else "needs_review",
        "device": "cpu",
        "architecture": "lykenox_compact_transposed_conv_v0",
        "ownership": "LYKENOX-owned PyTorch model; no external vocoder runtime",
        "utterance_id": str(selected["utterance_id"]),
        "wav_path": str(wav_path),
        "parameters": model.parameter_count(),
        "sample_rate": vocoder_config.sample_rate,
        "hop_length": vocoder_config.hop_length,
        "mel_bins": vocoder_config.mel_bins,
        "upsample_factors": list(vocoder_config.upsample_factors),
        "upsample_product": math.prod(vocoder_config.upsample_factors),
        "train_steps": train_steps,
        "train_mel_frames": train_mel_frames,
        "train_audio_seconds": round(expected_train_samples / vocoder_config.sample_rate, 4),
        "expected_train_waveform_samples": expected_train_samples,
        "exact_waveform_length": exact_length,
        "first_training_loss": round(losses[0], 6),
        "last_training_loss": round(losses[-1], 6),
        "loss_decreased": loss_decreased,
        "mean_training_seconds_per_step": round(statistics.fmean(timings), 4),
        "min_training_seconds_per_step": round(min(timings), 4),
        "max_training_seconds_per_step": round(max(timings), 4),
        "max_gradient_norm": round(max_grad_norm, 6),
        "inference": inference,
        "realtime_inference_pass": realtime,
        "next_gate": (
            "design_persistent_vocoder_training_contract"
            if gate_pass
            else "optimize_or_resize_lykenox_vocoder_cpu"
        ),
        "warning": (
            "This benchmark validates compute feasibility only. The short waveform + "
            "spectral objective is not a final vocoder training recipe and this model "
            "is not yet a perceptually validated runtime artifact."
        ),
    }
    artifact_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_cpu_benchmark"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--train-mel-frames", type=int, default=32)
    parser.add_argument("--inference-mel-frames", type=int, default=96)
    parser.add_argument("--inference-repeats", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_vocoder_cpu_benchmark(
                args.root,
                train_steps=args.train_steps,
                train_mel_frames=args.train_mel_frames,
                inference_mel_frames=args.inference_mel_frames,
                inference_repeats=args.inference_repeats,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
