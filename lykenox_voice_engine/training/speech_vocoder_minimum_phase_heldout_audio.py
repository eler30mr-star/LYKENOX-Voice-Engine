"""Render complete held-out utterances from the validation-selected directional checkpoint.

The checkpoint supplies the exact fixed loss weights used during training.  Output is written
as FLOAT WAV without gain normalization, EQ, denoising, duration modification or any other
post-hoc enhancement.  Metrics remain rejection diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from lykenox_voice_engine.training.speech_vocoder_minimum_phase_artifact_v2 import (
    load_minimum_phase_checkpoint_v2,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_directional_weight_calibration import (
    fixed_weights_from_mapping,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    FULL_UTTERANCE_DATA_VERSION,
    collect_owned_vocoder_utterances,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_noise import (
    NOISE_SEED_VERSION,
    stable_owned_noise_seed,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_objective_v3 import (
    ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
    OwnedMinimumPhaseObjectiveV3,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_renderer import (
    HOP_LENGTH,
    RENDERER_VERSION,
    SAMPLE_RATE,
    render_owned_minimum_phase_vocoder_path,
)


HELDOUT_AUDIO_VERSION = "owned-minimum-phase-heldout-audio-v3-directional-fixed"
POSTHOC_GAIN_NORMALIZATION_USED = False
POSTHOC_EQ_USED = False
POSTHOC_DENOISING_USED = False
PREDICTED_DURATION_MODIFIED = False
METRICS_ACCEPT_VOICE_QUALITY = False


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )


def _write_float_wav(path: Path, waveform: torch.Tensor) -> None:
    values = waveform.detach().cpu().to(torch.float32).numpy()
    sf.write(str(path), values, SAMPLE_RATE, subtype="FLOAT")


def render_heldout_audio(
    root: Path,
    *,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    split: str = "val",
    max_items: int = 5,
    noise_seed: int = 97,
) -> dict[str, object]:
    if split == "train":
        raise ValueError("held-out audio must not use the train split")
    if max_items < 1:
        raise ValueError("max_items must be positive")
    root = Path(root).resolve()
    checkpoint = (
        Path(checkpoint).resolve()
        if checkpoint is not None
        else root / "models" / "lykenox_identity" / "training" / "vocoder_minimum_phase_v3" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"minimum-phase best checkpoint does not exist: {checkpoint}")
    if checkpoint.name != "best.pt":
        raise ValueError("held-out listening requires validation-selected best.pt")
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "models" / "lykenox_identity" / "evaluation" / "vocoder_minimum_phase_v3_heldout"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_minimum_phase_checkpoint_v2(checkpoint)
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("directional checkpoint is missing run config")
    weights = fixed_weights_from_mapping(run_config["calibrated_weights"])
    model.eval()
    objective = OwnedMinimumPhaseObjectiveV3(weights).cpu()
    utterances = collect_owned_vocoder_utterances(root, split, max_items=max_items)

    items: list[dict[str, object]] = []
    with torch.no_grad():
        for utterance in utterances:
            mel = utterance.mel.unsqueeze(0).cpu()
            f0_hz = utterance.f0_hz.unsqueeze(0).cpu()
            voiced = utterance.voiced.unsqueeze(0).cpu()
            periodicity = utterance.periodicity.unsqueeze(0).cpu()
            target = utterance.waveform.unsqueeze(0).cpu()
            cepstrum = model(mel, f0_hz, voiced, periodicity)
            item_noise_seed = stable_owned_noise_seed(
                noise_seed,
                split=utterance.split,
                utterance_id=utterance.utterance_id,
                start_frame=0,
            )
            prediction, _ = render_owned_minimum_phase_vocoder_path(
                cepstrum,
                f0_hz,
                voiced,
                periodicity,
                noise_seed=item_noise_seed,
            )
            expected_samples = utterance.mel_frames * HOP_LENGTH
            if prediction.shape[-1] != expected_samples or target.shape[-1] != expected_samples:
                raise RuntimeError("held-out audio violated exact full-utterance length contract")
            losses = objective(prediction, target, mel)
            stem = _safe_name(utterance.utterance_id)
            prediction_path = output_dir / f"{stem}__prediction.wav"
            reference_path = output_dir / f"{stem}__reference.wav"
            _write_float_wav(prediction_path, prediction.squeeze(0))
            _write_float_wav(reference_path, target.squeeze(0))
            items.append(
                {
                    "utterance_id": utterance.utterance_id,
                    "mel_frames": utterance.mel_frames,
                    "samples": expected_samples,
                    "seconds": expected_samples / float(SAMPLE_RATE),
                    "noise_seed": item_noise_seed,
                    "prediction": str(prediction_path),
                    "reference": str(reference_path),
                    "diagnostic_loss": {
                        "total": float(losses.total),
                        **losses.detached_terms(),
                    },
                }
            )

    report = {
        "status": "ready_for_listening",
        "heldout_audio_version": HELDOUT_AUDIO_VERSION,
        "full_utterance_data_version": FULL_UTTERANCE_DATA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "objective_version": ACTIVE_MINIMUM_PHASE_OBJECTIVE_VERSION,
        "noise_seed_version": NOISE_SEED_VERSION,
        "base_noise_seed": int(noise_seed),
        "calibrated_weights": weights.as_dict(),
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_selection": "best_validation",
        "checkpoint_global_step": int(payload["progress"]["global_step"]),
        "split": split,
        "item_count": len(items),
        "items": items,
        "posthoc_gain_normalization_used": False,
        "posthoc_eq_used": False,
        "posthoc_denoising_used": False,
        "predicted_duration_modified": False,
        "metrics_accept_voice_quality": False,
        "product_acceptance_requires_human_listening": True,
    }
    _atomic_json(output_dir / "heldout_audio_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-items", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            render_heldout_audio(
                args.root,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                split=args.split,
                max_items=args.max_items,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
