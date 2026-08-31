"""Non-persistent product-chain smoke for the zero-init mel residual postnet.

The accepted acoustic-v2 and vocoder-v4.2 checkpoints remain immutable. Only the postnet
is trained, on one real training utterance with teacher durations. The gate requires exact
baseline identity before training, exact duration/F0/voicing preservation after training,
mel-fidelity improvement, and improved v4.2 waveform presence without frame-grid failure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from lykenox_voice_engine.models.speech.mel_postnet import (
    MEL_POSTNET_ARCHITECTURE_V1,
    LykenoxAcousticMelPostnetCandidate,
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
from lykenox_voice_engine.training.speech_vocoder_grid_artifact import frame_grid_artifact_metrics
from lykenox_voice_engine.training.speech_vocoder_level_presence_loss import target_relative_presence_loss
from lykenox_voice_engine.training.speech_vocoder_v4_2_artifact import load_v4_2_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v4_2_full_utterance_oracle_acceptance import (
    _load_reference_waveform,
)


SMOKE_VERSION = "acoustic-mel-residual-postnet-product-smoke-v1"
UPDATES = 24
LEARNING_RATE = 3e-4


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected(root: Path) -> dict[str, Path]:
    training = root / "models" / "lykenox_identity" / "training"
    return {
        "acoustic_v2_best": training / "acoustic_frame_context_v2" / "best.pt",
        "acoustic_mel_fidelity_best": training / "acoustic_mel_fidelity_v1" / "best.pt",
        "acoustic_mel_fidelity_last": training / "acoustic_mel_fidelity_v1" / "last.pt",
        "v4_2_best": training / "vocoder_source_filter_v4_2" / "best.pt",
        "v7_best": training / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def _loss(model_output, batch, *, sample_rate: int, n_fft: int):
    return acoustic_mel_fidelity_loss(
        model_output["mel"],
        batch.mel,
        batch.mel_mask,
        sample_rate=sample_rate,
        n_fft=n_fft,
    )


def run_mel_postnet_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    protected = _protected(root)
    if not protected["acoustic_v2_best"].exists() or not protected["v4_2_best"].exists():
        raise FileNotFoundError("accepted acoustic-v2 and v4.2 checkpoints are required")
    before = {name: _sha256(path) for name, path in protected.items()}

    base, _base_payload = load_acoustic_prosody_checkpoint(protected["acoustic_v2_best"])
    vocoder, _disc, _vocoder_payload = load_v4_2_checkpoint(protected["v4_2_best"])
    base.cpu().eval()
    vocoder.cpu().eval()
    candidate = LykenoxAcousticMelPostnetCandidate(base, hidden_channels=128).cpu()
    candidate.eval()
    trainable = candidate.trainable_parameter_names()
    if not trainable or any(not name.startswith("postnet.") for name in trainable):
        raise RuntimeError("mel postnet smoke may train only postnet parameters")

    duration_root = find_clean_duration_root(root)
    dataset = LykenoxAlignedSpeechDataset(
        root,
        "train",
        base.config,
        duration_root=duration_root,
        include_pitch_targets=True,
    )
    item = dataset[0]
    batch = collate_aligned_speech([item]).to("cpu")
    if batch.f0_hz is None or batch.voiced is None:
        raise RuntimeError("mel postnet smoke requires target F0/voicing")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with torch.no_grad():
        base_output = base(batch.token_ids, batch.token_mask, batch.durations)
        initial_output = candidate(batch.token_ids, batch.token_mask, batch.durations)
    zero_init_mel_exact = torch.equal(initial_output["mel"], base_output["mel"])
    duration_exact_initial = torch.equal(
        initial_output["duration_prediction"], base_output["duration_prediction"]
    ) and torch.equal(initial_output["regulated_durations"], base_output["regulated_durations"])
    prosody_exact_initial = torch.equal(
        initial_output["f0_prediction_hz"], base_output["f0_prediction_hz"]
    ) and torch.equal(initial_output["voicing_logits"], base_output["voicing_logits"])
    if not (zero_init_mel_exact and duration_exact_initial and prosody_exact_initial):
        raise RuntimeError("zero-init postnet did not preserve accepted acoustic baseline")

    frames = int(batch.mel_lengths[0])
    samples = frames * base.config.hop_length
    reference = _load_reference_waveform(
        Path(str(dataset.base[0]["wav_path"])),
        sample_rate=base.config.sample_rate,
        samples=samples,
    )
    centers = _mel_bin_centers_hz(
        sample_rate=base.config.sample_rate,
        n_fft=base.config.n_fft,
        mel_bins=base.config.mel_bins,
    )
    initial_metrics = _mel_metrics(
        base_output["mel"][0, :frames], batch.mel[0, :frames], centers_hz=centers
    )
    initial_loss_result = _loss(
        initial_output,
        batch,
        sample_rate=base.config.sample_rate,
        n_fft=base.config.n_fft,
    )
    initial_total = float(initial_loss_result.total.detach())

    with torch.no_grad():
        base_wave = vocoder(
            base_output["mel"][:, :frames],
            batch.f0_hz[:, :frames],
            batch.voiced[:, :frames],
        )[0]
        base_presence = target_relative_presence_loss(
            base_wave.unsqueeze(0),
            reference.unsqueeze(0),
            sample_rate=base.config.sample_rate,
            hop_length=base.config.hop_length,
        )

    optimizer = torch.optim.AdamW(candidate.postnet.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    finite_gradients = True
    best_total = initial_total
    best_state = {k: v.detach().clone() for k, v in candidate.postnet.state_dict().items()}
    for _ in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        output = candidate(batch.token_ids, batch.token_mask, batch.durations)
        result = _loss(output, batch, sample_rate=base.config.sample_rate, n_fft=base.config.n_fft)
        if not torch.isfinite(result.total):
            raise RuntimeError("non-finite mel postnet smoke loss")
        result.total.backward()
        for parameter in candidate.postnet.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                finite_gradients = False
        if not finite_gradients:
            raise RuntimeError("non-finite mel postnet smoke gradients")
        torch.nn.utils.clip_grad_norm_(candidate.postnet.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            post = candidate(batch.token_ids, batch.token_mask, batch.durations)
            post_result = _loss(
                post,
                batch,
                sample_rate=base.config.sample_rate,
                n_fft=base.config.n_fft,
            )
        value = float(post_result.total.detach())
        if value < best_total:
            best_total = value
            best_state = {k: v.detach().clone() for k, v in candidate.postnet.state_dict().items()}

    candidate.postnet.load_state_dict(best_state)
    with torch.no_grad():
        final_output = candidate(batch.token_ids, batch.token_mask, batch.durations)
        final_result = _loss(
            final_output,
            batch,
            sample_rate=base.config.sample_rate,
            n_fft=base.config.n_fft,
        )
        refined_wave = vocoder(
            final_output["mel"][:, :frames],
            batch.f0_hz[:, :frames],
            batch.voiced[:, :frames],
        )[0]
        refined_presence = target_relative_presence_loss(
            refined_wave.unsqueeze(0),
            reference.unsqueeze(0),
            sample_rate=base.config.sample_rate,
            hop_length=base.config.hop_length,
        )

    final_metrics = _mel_metrics(
        final_output["mel"][0, :frames], batch.mel[0, :frames], centers_hz=centers
    )
    duration_exact_final = torch.equal(
        final_output["duration_prediction"], base_output["duration_prediction"]
    ) and torch.equal(final_output["regulated_durations"], base_output["regulated_durations"])
    prosody_exact_final = torch.equal(
        final_output["f0_prediction_hz"], base_output["f0_prediction_hz"]
    ) and torch.equal(final_output["voicing_logits"], base_output["voicing_logits"])

    base_grid = frame_grid_artifact_metrics(
        base_wave,
        sample_rate=base.config.sample_rate,
        hop_length=base.config.hop_length,
    )
    refined_grid = frame_grid_artifact_metrics(
        refined_wave,
        sample_rate=base.config.sample_rate,
        hop_length=base.config.hop_length,
    )
    final_total = float(final_result.total.detach())
    total_decreased = final_total < initial_total
    mel_l1_decreased = float(final_result.mel_l1.detach()) < float(initial_loss_result.mel_l1.detach())
    spectral_ratio_closer = abs(final_metrics["spectral_delta_ratio"] - 1.0) < abs(
        initial_metrics["spectral_delta_ratio"] - 1.0
    )
    temporal_ratio_closer = abs(final_metrics["temporal_delta_ratio"] - 1.0) < abs(
        initial_metrics["temporal_delta_ratio"] - 1.0
    )
    high_band_closer = abs(final_metrics["band_3k_8k_relative_db"]) < abs(
        initial_metrics["band_3k_8k_relative_db"]
    )
    base_presence_error = abs(float(base_presence.presence_1k_8k_error_db.detach()))
    refined_presence_error = abs(float(refined_presence.presence_1k_8k_error_db.detach()))
    waveform_presence_closer = refined_presence_error < base_presence_error
    no_new_grid_failure = not bool(refined_grid.severe_grid_artifact[0])

    after = {name: _sha256(path) for name, path in protected.items()}
    protected_unchanged = before == after
    status = "pass" if all((
        zero_init_mel_exact,
        duration_exact_initial,
        prosody_exact_initial,
        duration_exact_final,
        prosody_exact_final,
        finite_gradients,
        total_decreased,
        mel_l1_decreased,
        spectral_ratio_closer,
        temporal_ratio_closer,
        high_band_closer,
        waveform_presence_closer,
        no_new_grid_failure,
        protected_unchanged,
    )) else "fail"
    return {
        "status": status,
        "smoke_version": SMOKE_VERSION,
        "architecture": MEL_POSTNET_ARCHITECTURE_V1,
        "loss_version": ACOUSTIC_MEL_FIDELITY_LOSS_VERSION,
        "updates": UPDATES,
        "learning_rate": LEARNING_RATE,
        "only_postnet_trainable": True,
        "zero_init_mel_exact": zero_init_mel_exact,
        "duration_outputs_exact": duration_exact_final,
        "f0_voicing_outputs_exact": prosody_exact_final,
        "finite_gradients": finite_gradients,
        "initial_total": round(initial_total, 6),
        "final_total": round(final_total, 6),
        "mel_l1_decreased": mel_l1_decreased,
        "spectral_delta_ratio": f"{initial_metrics['spectral_delta_ratio']:.6f} -> {final_metrics['spectral_delta_ratio']:.6f}",
        "temporal_delta_ratio": f"{initial_metrics['temporal_delta_ratio']:.6f} -> {final_metrics['temporal_delta_ratio']:.6f}",
        "band_3k_8k_relative_db": f"{initial_metrics['band_3k_8k_relative_db']:.6f} -> {final_metrics['band_3k_8k_relative_db']:.6f}",
        "v4_2_presence_1k_8k_error_db": f"{base_presence_error:.6f} -> {refined_presence_error:.6f}",
        "waveform_presence_closer": waveform_presence_closer,
        "base_grid_failure": bool(base_grid.severe_grid_artifact[0]),
        "refined_grid_failure": bool(refined_grid.severe_grid_artifact[0]),
        "protected_checkpoints_unchanged": protected_unchanged,
        "persistent_training_started": False,
        "training_authorized": False,
        "next_gate": (
            "build_exact_resume_mel_postnet_candidate" if status == "pass"
            else "reject_mel_postnet_before_persistent_training"
        ),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(run_mel_postnet_smoke(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
