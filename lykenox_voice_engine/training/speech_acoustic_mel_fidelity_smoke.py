"""Non-persistent micro-overfit gate for isolated acoustic mel fidelity refinement.

Loads the accepted acoustic frame-context-v2 checkpoint, freezes every parameter except
``mel_decoder``, and performs a tiny teacher-duration micro-overfit in memory.  The gate
must improve the explicit mel-fidelity objective while duration/F0/voicing outputs and all
persistent checkpoints remain bit-exact.  No artifact is saved and no persistent training
is authorized here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lykenox_voice_engine.training.speech_acoustic_frame_context_train import (
    FRAME_CONTEXT_VERSION,
    TRAINER_CONTRACT_VERSION,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_audit import (
    _mel_bin_centers_hz,
    _mel_metrics,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_loss import (
    ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
    acoustic_mel_fidelity_loss,
)
from lykenox_voice_engine.training.speech_acoustic_prosody_artifact import (
    load_acoustic_prosody_checkpoint,
)
from lykenox_voice_engine.training.speech_aligned_data import (
    LykenoxAlignedSpeechDataset,
    collate_aligned_speech,
    find_clean_duration_root,
)


SMOKE_VERSION = "acoustic-mel-fidelity-isolated-smoke-v1"
UPDATES = 12
LEARNING_RATE = 5e-4


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_paths(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "acoustic_v2_last": training / "acoustic_frame_context_v2" / "last.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_best": training / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_best": training / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def _loss_payload(result) -> dict[str, float]:
    return {
        "total": float(result.total.detach()),
        "mel_l1": float(result.mel_l1.detach()),
        "centered_shape": float(result.centered_shape.detach()),
        "spectral_delta": float(result.spectral_delta.detach()),
        "temporal_delta": float(result.temporal_delta.detach()),
        "clarity_underpresence": float(result.clarity_underpresence.detach()),
    }


def run_acoustic_mel_fidelity_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected_paths(root)
    if not protected["acoustic_v2_best"].exists():
        raise FileNotFoundError("Accepted acoustic v2 best checkpoint is required")
    if not protected["v4_2_best"].exists():
        raise FileNotFoundError("Accepted v4.2 best checkpoint is required")
    before_hashes = {name: _sha256(path) for name, path in protected.items()}

    model, payload = load_acoustic_prosody_checkpoint(protected["acoustic_v2_best"])
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise RuntimeError("Accepted acoustic v2 checkpoint is missing run_config")
    identity_exact = (
        model.config.frame_context_version == FRAME_CONTEXT_VERSION
        and run_config.get("trainer_contract_version") == TRAINER_CONTRACT_VERSION
        and run_config.get("frame_context_version") == FRAME_CONTEXT_VERSION
    )
    if not identity_exact:
        raise RuntimeError("Mel fidelity smoke requires accepted acoustic frame-context v2")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        model.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    if len(dataset) < 1:
        raise RuntimeError("Training dataset is empty")
    batch = collate_aligned_speech([dataset[0]]).to("cpu")
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("Mel fidelity smoke requires cached pitch targets for invariants")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model.cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mel_decoder.parameters():
        parameter.requires_grad_(True)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names or any(not name.startswith("mel_decoder.") for name in trainable_names):
        raise RuntimeError("Only mel_decoder parameters may be trainable")

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("mel_decoder.")
    }
    with torch.no_grad():
        baseline_output = model(batch.token_ids, batch.token_mask, batch.durations)
    if not torch.equal(baseline_output["regulated_durations"], batch.durations):
        raise RuntimeError("Teacher duration contract changed before smoke")

    centers = _mel_bin_centers_hz(
        sample_rate=model.config.sample_rate,
        n_fft=model.config.n_fft,
        mel_bins=model.config.mel_bins,
    )
    frames = int(batch.mel_lengths[0])
    initial_metrics = _mel_metrics(
        baseline_output["mel"][0, :frames],
        batch.mel[0, :frames],
        centers_hz=centers,
    )
    initial_result = acoustic_mel_fidelity_loss(
        baseline_output["mel"],
        batch.mel,
        batch.mel_mask,
        sample_rate=model.config.sample_rate,
        n_fft=model.config.n_fft,
    )
    initial_loss = _loss_payload(initial_result)

    optimizer = torch.optim.AdamW(
        model.mel_decoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    best_total = initial_loss["total"]
    best_state = {
        key: value.detach().clone()
        for key, value in model.mel_decoder.state_dict().items()
    }
    best_loss = dict(initial_loss)
    finite_gradients = True
    for _update in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.token_ids, batch.token_mask, batch.durations)
        result = acoustic_mel_fidelity_loss(
            output["mel"],
            batch.mel,
            batch.mel_mask,
            sample_rate=model.config.sample_rate,
            n_fft=model.config.n_fft,
        )
        if not torch.isfinite(result.total):
            raise RuntimeError("Non-finite mel fidelity smoke loss")
        result.total.backward()
        for parameter in model.mel_decoder.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                finite_gradients = False
        if not finite_gradients:
            raise RuntimeError("Non-finite mel fidelity smoke gradient")
        optimizer.step()

        with torch.no_grad():
            post_output = model(batch.token_ids, batch.token_mask, batch.durations)
            post_result = acoustic_mel_fidelity_loss(
                post_output["mel"],
                batch.mel,
                batch.mel_mask,
                sample_rate=model.config.sample_rate,
                n_fft=model.config.n_fft,
            )
        current = _loss_payload(post_result)
        if current["total"] < best_total:
            best_total = current["total"]
            best_loss = current
            best_state = {
                key: value.detach().clone()
                for key, value in model.mel_decoder.state_dict().items()
            }

    model.mel_decoder.load_state_dict(best_state)
    with torch.no_grad():
        final_output = model(batch.token_ids, batch.token_mask, batch.durations)
        final_result = acoustic_mel_fidelity_loss(
            final_output["mel"],
            batch.mel,
            batch.mel_mask,
            sample_rate=model.config.sample_rate,
            n_fft=model.config.n_fft,
        )
    final_loss = _loss_payload(final_result)
    final_metrics = _mel_metrics(
        final_output["mel"][0, :frames],
        batch.mel[0, :frames],
        centers_hz=centers,
    )

    frozen_parameters_exact = all(
        torch.equal(parameter.detach(), frozen_before[name])
        for name, parameter in model.named_parameters()
        if name in frozen_before
    )
    duration_prediction_exact = torch.equal(
        final_output["duration_prediction"], baseline_output["duration_prediction"]
    )
    regulated_durations_exact = torch.equal(
        final_output["regulated_durations"], baseline_output["regulated_durations"]
    )
    f0_prediction_exact = torch.equal(
        final_output["f0_prediction_hz"], baseline_output["f0_prediction_hz"]
    )
    voicing_logits_exact = torch.equal(
        final_output["voicing_logits"], baseline_output["voicing_logits"]
    )

    total_decreased = final_loss["total"] < initial_loss["total"]
    mel_l1_decreased = final_loss["mel_l1"] < initial_loss["mel_l1"]
    centered_shape_decreased = final_loss["centered_shape"] <= initial_loss["centered_shape"]
    spectral_delta_decreased = final_loss["spectral_delta"] <= initial_loss["spectral_delta"]
    temporal_delta_decreased = final_loss["temporal_delta"] <= initial_loss["temporal_delta"]
    clarity_guard_decreased = (
        final_loss["clarity_underpresence"] <= initial_loss["clarity_underpresence"]
    )

    # Ratio/dB movement is useful diagnostic evidence but is not substituted for the actual
    # optimized objective components in the trainability gate.
    spectral_ratio_closer = abs(final_metrics["spectral_delta_ratio"] - 1.0) < abs(
        initial_metrics["spectral_delta_ratio"] - 1.0
    )
    temporal_ratio_closer = abs(final_metrics["temporal_delta_ratio"] - 1.0) < abs(
        initial_metrics["temporal_delta_ratio"] - 1.0
    )
    high_band_closer = abs(final_metrics["band_3k_8k_relative_db"]) < abs(
        initial_metrics["band_3k_8k_relative_db"]
    )

    after_hashes = {name: _sha256(path) for name, path in protected.items()}
    protected_checkpoints_unchanged = before_hashes == after_hashes
    isolation_pass = all(
        (
            frozen_parameters_exact,
            duration_prediction_exact,
            regulated_durations_exact,
            f0_prediction_exact,
            voicing_logits_exact,
            protected_checkpoints_unchanged,
        )
    )
    trainability_pass = all(
        (
            finite_gradients,
            total_decreased,
            mel_l1_decreased,
            centered_shape_decreased,
            spectral_delta_decreased,
            temporal_delta_decreased,
            clarity_guard_decreased,
        )
    )
    status = "pass" if identity_exact and isolation_pass and trainability_pass else "fail"
    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "loss_version": ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
        "updates": UPDATES,
        "learning_rate": LEARNING_RATE,
        "acoustic_identity_exact": identity_exact,
        "trainable_parameter_names": trainable_names,
        "only_mel_decoder_trainable": all(
            name.startswith("mel_decoder.") for name in trainable_names
        ),
        "frozen_parameters_exact": frozen_parameters_exact,
        "duration_prediction_exact": duration_prediction_exact,
        "regulated_durations_exact": regulated_durations_exact,
        "f0_prediction_exact": f0_prediction_exact,
        "voicing_logits_exact": voicing_logits_exact,
        "finite_gradients": finite_gradients,
        "initial_loss": {key: round(value, 6) for key, value in initial_loss.items()},
        "best_loss_during_updates": {
            key: round(value, 6) for key, value in best_loss.items()
        },
        "final_loss": {key: round(value, 6) for key, value in final_loss.items()},
        "initial_fidelity_metrics": {
            key: round(float(value), 6) for key, value in initial_metrics.items()
        },
        "final_fidelity_metrics": {
            key: round(float(value), 6) for key, value in final_metrics.items()
        },
        "total_decreased": total_decreased,
        "mel_l1_decreased": mel_l1_decreased,
        "centered_shape_decreased": centered_shape_decreased,
        "spectral_delta_decreased": spectral_delta_decreased,
        "temporal_delta_decreased": temporal_delta_decreased,
        "clarity_guard_decreased": clarity_guard_decreased,
        "spectral_ratio_closer_to_target": spectral_ratio_closer,
        "temporal_ratio_closer_to_target": temporal_ratio_closer,
        "band_3k_8k_closer_to_target": high_band_closer,
        "isolation_pass": isolation_pass,
        "trainability_pass": trainability_pass,
        "protected_checkpoints_unchanged": protected_checkpoints_unchanged,
        "persistent_training_started": False,
        "training_authorized": False,
        "vocoder_modified": False,
        "predicted_duration_modified": False,
        "next_gate": (
            "build_exact_resume_isolated_mel_decoder_candidate"
            if status == "pass"
            else "revise_mel_fidelity_objective_before_persistent_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_acoustic_mel_fidelity_smoke(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
