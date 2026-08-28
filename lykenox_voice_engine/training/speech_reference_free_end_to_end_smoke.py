"""First reference-free LYKENOX Speech text -> waveform integration gate.

This is not a training run.  It connects the accepted persistent acoustic frame-context v2
checkpoint to the accepted persistent v4.1 source-filter vocoder using only text-derived
predictions.  No reference WAV, source speaker, waveform pitch target, or voice conversion
path is accepted by this gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import soundfile as sf
import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech.config import FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
from lykenox_voice_engine.models.speech.duration_policy import PREDICTED_DURATION_POLICY_VERSION
from lykenox_voice_engine.models.vocoder.network_v4_1 import VOCODER_GENERATOR_V4_1_ARCHITECTURE
from lykenox_voice_engine.runtime.speech_conditioning import (
    PREDICTED_SPEECH_F0_MAX_HZ,
    PREDICTED_SPEECH_F0_MIN_HZ,
    SPEECH_VOCODER_CONDITIONING_VERSION,
    prepare_speech_vocoder_conditioning,
)
from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    TRAINER_CONTRACT_VERSION as ACOUSTIC_TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_source_filter_artifact import (
    load_source_filter_checkpoint,
)


SMOKE_VERSION = "reference-free-text-to-waveform-smoke-v1"

PROBE_TEXTS = (
    "La voz de Lykenox debe conservar un ritmo natural y estable.",
    "Esta prueba usa únicamente texto para predecir duración, tono y sonoridad.",
    "Hola mundo, hoy comprobamos pausas, palabras y una frase un poco más larga.",
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _band_fractions(waveform: torch.Tensor, sample_rate: int) -> dict[str, float]:
    waveform = waveform.to(torch.float64)
    if waveform.numel() < 2:
        return {"0_80": 0.0, "80_300": 0.0, "300_3000": 0.0, "3000_nyquist": 0.0}
    window = torch.hann_window(waveform.numel(), periodic=False, dtype=waveform.dtype)
    spectrum = torch.fft.rfft(waveform * window)
    power = spectrum.abs().square()
    frequencies = torch.fft.rfftfreq(waveform.numel(), d=1.0 / float(sample_rate))
    total = float(power.sum())
    if not math.isfinite(total) or total <= 1e-20:
        return {"0_80": 0.0, "80_300": 0.0, "300_3000": 0.0, "3000_nyquist": 0.0}

    def fraction(low: float, high: float | None) -> float:
        mask = frequencies >= low
        if high is not None:
            mask = mask & (frequencies < high)
        return float(power[mask].sum()) / total

    return {
        "0_80": fraction(0.0, 80.0),
        "80_300": fraction(80.0, 300.0),
        "300_3000": fraction(300.0, 3000.0),
        "3000_nyquist": fraction(3000.0, None),
    }


def run_reference_free_end_to_end_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    acoustic_checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "acoustic_frame_context_v2"
        / "best.pt"
    )
    vocoder_checkpoint = (
        root
        / "models"
        / "lykenox_identity"
        / "training"
        / "vocoder_source_filter_v4_1"
        / "best.pt"
    )
    if not acoustic_checkpoint.exists():
        raise FileNotFoundError(f"Accepted acoustic v2 checkpoint not found: {acoustic_checkpoint}")
    if not vocoder_checkpoint.exists():
        raise FileNotFoundError(f"Accepted v4.1 vocoder checkpoint not found: {vocoder_checkpoint}")

    output_dir = (
        root
        / "models"
        / "lykenox_identity"
        / "evaluation"
        / "reference_free_speech_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "reference_free_smoke_report.json"

    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    acoustic, acoustic_payload = load_acoustic_prosody_checkpoint(acoustic_checkpoint)
    acoustic_run = acoustic_payload.get("run_config")
    if not isinstance(acoustic_run, dict):
        raise RuntimeError("Accepted acoustic v2 checkpoint is missing run_config")
    acoustic_identity_exact = (
        acoustic.config.frame_context_version == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
        and acoustic_run.get("trainer_contract_version") == ACOUSTIC_TRAINER_CONTRACT_VERSION
        and acoustic_run.get("frame_context_version") == FRAME_CONTEXT_TOKEN_PROGRESS_CONV_V1
    )
    if not acoustic_identity_exact:
        raise RuntimeError("Reference-free smoke requires the accepted acoustic frame-context v2 identity")

    vocoder, _discriminator, vocoder_payload = load_source_filter_checkpoint(vocoder_checkpoint)
    vocoder_identity_exact = (
        vocoder_payload.get("generator_architecture") == VOCODER_GENERATOR_V4_1_ARCHITECTURE
        and vocoder.architecture == VOCODER_GENERATOR_V4_1_ARCHITECTURE
    )
    if not vocoder_identity_exact:
        raise RuntimeError("Reference-free smoke requires the accepted persistent v4.1 vocoder")

    acoustic_vocoder_contract_exact = (
        acoustic.config.mel_bins == vocoder.config.mel_bins
        and acoustic.config.sample_rate == vocoder.config.sample_rate
        and acoustic.config.hop_length == vocoder.config.hop_length
    )
    if not acoustic_vocoder_contract_exact:
        raise RuntimeError("Acoustic/vocoder mel, sample-rate, or hop contract mismatch")

    acoustic.cpu().eval()
    vocoder.cpu().eval()
    frontend = SpanishTextFrontend()

    probes: list[dict[str, object]] = []
    all_finite = True
    all_length_exact = True
    all_wav_headers_exact = True
    all_non_silent = True
    all_have_voiced_frames = True

    with torch.inference_mode():
        for index, text in enumerate(PROBE_TEXTS, start=1):
            processed = frontend.process(text)
            token_ids = torch.tensor([processed.token_ids], dtype=torch.long)
            token_mask = torch.ones_like(token_ids, dtype=torch.bool)
            acoustic_output = acoustic(token_ids, token_mask)
            conditioning = prepare_speech_vocoder_conditioning(acoustic_output)

            frame_count = int(acoustic_output["mel_lengths"][0])
            duration_sum = int(acoustic_output["regulated_durations"].sum())
            mel = conditioning.mel[:, :frame_count]
            f0_hz = conditioning.f0_hz[:, :frame_count]
            voiced = conditioning.voiced[:, :frame_count]
            raw_f0 = conditioning.raw_f0_hz[:, :frame_count]
            clipped_mask = conditioning.f0_clipped_mask[:, :frame_count]

            frame_contract_exact = (
                frame_count == duration_sum
                and mel.shape[1] == frame_count
                and f0_hz.shape[1] == frame_count
                and voiced.shape[1] == frame_count
            )
            waveform = vocoder(mel, f0_hz, voiced)
            expected_samples = frame_count * acoustic.config.hop_length
            waveform_length_exact = tuple(waveform.shape) == (1, expected_samples)
            wave = waveform[0].detach().cpu().to(torch.float32).contiguous()

            finite = bool(
                torch.isfinite(mel).all()
                and torch.isfinite(f0_hz).all()
                and torch.isfinite(voiced).all()
                and torch.isfinite(wave).all()
            )
            rms = float(torch.sqrt(torch.mean(wave.square())))
            peak = float(wave.abs().max())
            non_silent = math.isfinite(rms) and math.isfinite(peak) and rms > 1e-5 and peak > 1e-4

            voiced_mask = voiced[0] > 0.5
            voiced_count = int(voiced_mask.sum())
            voiced_fraction = voiced_count / max(1, frame_count)
            has_voiced = voiced_count > 0
            if has_voiced:
                active_f0 = f0_hz[0, voiced_mask]
                raw_active_f0 = raw_f0[0, voiced_mask]
                f0_min = float(active_f0.min())
                f0_max = float(active_f0.max())
                raw_f0_min = float(raw_active_f0.min())
                raw_f0_max = float(raw_active_f0.max())
                clipped_fraction = float(clipped_mask[0, voiced_mask].float().mean())
            else:
                f0_min = f0_max = raw_f0_min = raw_f0_max = None
                clipped_fraction = None

            wav_path = output_dir / f"{index:02d}_reference_free.wav"
            sf.write(
                str(wav_path),
                wave.numpy(),
                acoustic.config.sample_rate,
                subtype="PCM_16",
            )
            info = sf.info(str(wav_path))
            wav_header_exact = (
                int(info.samplerate) == acoustic.config.sample_rate
                and int(info.frames) == expected_samples
                and int(info.channels) == 1
            )

            all_finite = all_finite and finite
            all_length_exact = all_length_exact and frame_contract_exact and waveform_length_exact
            all_wav_headers_exact = all_wav_headers_exact and wav_header_exact
            all_non_silent = all_non_silent and non_silent
            all_have_voiced_frames = all_have_voiced_frames and has_voiced

            probes.append(
                {
                    "index": index,
                    "text": text,
                    "normalized_text": processed.normalized_text,
                    "token_count": len(processed.token_ids),
                    "mel_frames": frame_count,
                    "regulated_duration_sum": duration_sum,
                    "frame_contract_exact": frame_contract_exact,
                    "waveform_samples": int(wave.numel()),
                    "expected_waveform_samples": expected_samples,
                    "waveform_length_exact": waveform_length_exact,
                    "sample_rate": acoustic.config.sample_rate,
                    "duration_seconds": round(expected_samples / acoustic.config.sample_rate, 4),
                    "finite_outputs": finite,
                    "rms": round(rms, 7),
                    "peak": round(peak, 7),
                    "non_silent": non_silent,
                    "predicted_voiced_fraction": round(voiced_fraction, 6),
                    "predicted_voiced_frame_count": voiced_count,
                    "predicted_f0_min_hz": None if f0_min is None else round(f0_min, 4),
                    "predicted_f0_max_hz": None if f0_max is None else round(f0_max, 4),
                    "raw_predicted_f0_min_hz_on_voiced": (
                        None if raw_f0_min is None else round(raw_f0_min, 4)
                    ),
                    "raw_predicted_f0_max_hz_on_voiced": (
                        None if raw_f0_max is None else round(raw_f0_max, 4)
                    ),
                    "voiced_f0_clipped_fraction": (
                        None if clipped_fraction is None else round(clipped_fraction, 6)
                    ),
                    "band_power_fraction": {
                        key: round(value, 6)
                        for key, value in _band_fractions(wave, acoustic.config.sample_rate).items()
                    },
                    "wav_header_exact": wav_header_exact,
                    "wav_path": str(wav_path),
                }
            )

    checks = {
        "acoustic_identity_exact": acoustic_identity_exact,
        "vocoder_identity_exact": vocoder_identity_exact,
        "acoustic_vocoder_contract_exact": acoustic_vocoder_contract_exact,
        "all_text_only_outputs_finite": all_finite,
        "all_duration_mel_waveform_lengths_exact": all_length_exact,
        "all_wav_headers_exact": all_wav_headers_exact,
        "all_waveforms_non_silent": all_non_silent,
        "all_probes_have_predicted_voiced_frames": all_have_voiced_frames,
    }
    status = "pass" if all(checks.values()) else "needs_review"
    report: dict[str, object] = {
        "status": status,
        "device": "cpu",
        "smoke_version": SMOKE_VERSION,
        "frontend_version": frontend.version,
        "acoustic_checkpoint": str(acoustic_checkpoint),
        "acoustic_trainer_contract_version": ACOUSTIC_TRAINER_CONTRACT_VERSION,
        "frame_context_version": acoustic.config.frame_context_version,
        "predicted_duration_policy_version": PREDICTED_DURATION_POLICY_VERSION,
        "speech_vocoder_conditioning_version": SPEECH_VOCODER_CONDITIONING_VERSION,
        "vocoder_checkpoint": str(vocoder_checkpoint),
        "vocoder_architecture": vocoder.architecture,
        "predicted_speech_f0_min_hz": PREDICTED_SPEECH_F0_MIN_HZ,
        "predicted_speech_f0_max_hz": PREDICTED_SPEECH_F0_MAX_HZ,
        **checks,
        "reference_audio_required": False,
        "waveform_pitch_target_required": False,
        "source_speaker_or_singer_required": False,
        "voice_conversion_required": False,
        "probe_count": len(probes),
        "probes": probes,
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "next_gate": (
            "listen_and_audit_reference_free_end_to_end_wavs"
            if status == "pass"
            else "fix_reference_free_acoustic_vocoder_integration"
        ),
        "warning": (
            "A pass proves the first complete reference-free text-to-waveform execution and "
            "exact integration contracts. Human/perceptual listening is still mandatory "
            "before this path is accepted for runtime/export."
        ),
    }
    _atomic_json(report_path, report)
    return {**report, "report_path": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        json.dumps(
            run_reference_free_end_to_end_smoke(args.root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
