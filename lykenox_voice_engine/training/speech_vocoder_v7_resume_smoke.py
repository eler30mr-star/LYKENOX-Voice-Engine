"""Exact-resume and hard epoch-two gate for V7 first-epoch training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from lykenox_voice_engine.training.speech_vocoder_v7_train import TRAINER_CONTRACT_VERSION, run_bounded_resumable_v7_first_epoch

SMOKE_VERSION = "vocoder-v7-first-epoch-exact-resume-smoke-v2"


def _sha(path: Path) -> str | None:
    if not path.exists(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def _exact(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor): return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict): return a.keys() == b.keys() and all(_exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _payload(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict): raise RuntimeError(f"Invalid v7 smoke checkpoint: {path}")
    return value


def _protected(root: Path) -> dict[str, Path]:
    base = root / "models" / "lykenox_identity" / "training"
    return {
        "v4_2": base / "vocoder_source_filter_v4_2" / "best.pt",
        "v6_prior_last": base / "vocoder_direct_waveform_v6" / "last.pt",
        "v6_prior_best": base / "vocoder_direct_waveform_v6" / "best.pt",
        "v6_clarity_last": base / "vocoder_direct_waveform_v6_clarity_guard_v1" / "last.pt",
        "v6_clarity_best": base / "vocoder_direct_waveform_v6_clarity_guard_v1" / "best.pt",
        "v7_persistent_last": base / "vocoder_source_free_v7_first_epoch" / "last.pt",
        "v7_persistent_best": base / "vocoder_source_free_v7_first_epoch" / "best.pt",
    }


def run_v7_resume_smoke(root: Path) -> dict[str, object]:
    root = Path(root).resolve(); torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    protected = _protected(root); before = {k: _sha(v) for k, v in protected.items()}
    common = dict(segment_mel_frames=32, train_items=6, val_items=2, seed=77123, generator_lr=2e-4, content_weight=0.75, level_weight=0.25, checkpoint_every_updates=20, time_budget_seconds=600.0, checkpoint_reserve_seconds=1.0, validation_reserve_seconds=1.0)
    with tempfile.TemporaryDirectory(prefix="lykenox-v7-resume-") as temporary:
        tmp = Path(temporary); direct = tmp / "direct"; split = tmp / "split"; gate = tmp / "gate"
        direct_result = run_bounded_resumable_v7_first_epoch(root, **common, max_updates_this_run=4, artifact_dir_override=direct)
        split_a = run_bounded_resumable_v7_first_epoch(root, **common, max_updates_this_run=2, artifact_dir_override=split)
        split_b = run_bounded_resumable_v7_first_epoch(root, **common, max_updates_this_run=2, artifact_dir_override=split)
        a = _payload(direct / "last.pt"); b = _payload(split / "last.pt"); meta_a, meta_b = a.get("training_metadata"), b.get("training_metadata")
        if not isinstance(meta_a, dict) or not isinstance(meta_b, dict): raise RuntimeError("v7 smoke metadata missing")
        checks = {
            "global_step_exact": a.get("global_step") == b.get("global_step") == 4,
            "epoch_exact": a.get("epoch") == b.get("epoch") == 1,
            "next_item_offset_exact": a.get("next_item_offset") == b.get("next_item_offset") == 4,
            "generator_state_exact": _exact(a.get("generator_state"), b.get("generator_state")),
            "generator_optimizer_exact": _exact(a.get("generator_optimizer_state"), b.get("generator_optimizer_state")),
            "torch_rng_state_exact": _exact(a.get("torch_rng_state"), b.get("torch_rng_state")),
            "run_config_exact": _exact(meta_a.get("run_config"), meta_b.get("run_config")),
            "architecture_exact": a.get("generator_architecture") == b.get("generator_architecture") == "lykenox_source_free_mel_latent_waveform_v7",
            "source_free_exact": a.get("source_free") is True and b.get("source_free") is True,
            "no_sample_phase_exact": a.get("sample_phase_conditioning") is False and b.get("sample_phase_conditioning") is False,
            "no_sample_pitch_exact": a.get("sample_rate_pitch_features") is False and b.get("sample_rate_pitch_features") is False,
            "no_noise_conditioning_exact": a.get("deterministic_noise_conditioning") is False and b.get("deterministic_noise_conditioning") is False,
            "no_level_rescue_exact": a.get("level_rescue_branch") is False and b.get("level_rescue_branch") is False,
        }
        gate_config = {**common, "train_items": 2, "seed": 77234}
        gate_first = run_bounded_resumable_v7_first_epoch(root, **gate_config, artifact_dir_override=gate)
        gate_checkpoint = gate / "last.pt"; gate_sha_before = _sha(gate_checkpoint); gate_second = run_bounded_resumable_v7_first_epoch(root, **gate_config, artifact_dir_override=gate); gate_sha_after = _sha(gate_checkpoint)
        epoch2_blocked = gate_first.get("status") == "gate_reached" and gate_first.get("epochs_completed") == 1 and gate_first.get("next_item_offset") == 0 and gate_second.get("status") == "gate_reached" and gate_second.get("global_step") == gate_first.get("global_step") and gate_sha_before == gate_sha_after
        temporary_artifacts_present_during_smoke = all((p / "last.pt").exists() for p in (direct, split, gate))
    after = {k: _sha(v) for k, v in protected.items()}; protected_unchanged = before == after
    status = "pass" if all(checks.values()) and protected_unchanged and epoch2_blocked else "fail"
    return {"status": status, "smoke_version": SMOKE_VERSION, "trainer_contract_version": TRAINER_CONTRACT_VERSION, "updates_compared": 4, **checks, "direct_stop_reason": direct_result.get("stop_reason"), "split_first_stop_reason": split_a.get("stop_reason"), "split_second_stop_reason": split_b.get("stop_reason"), "epoch1_gate_reached": gate_first.get("status") == "gate_reached", "epoch2_training_blocked": epoch2_blocked, "gate_checkpoint_unchanged_on_rerun": gate_sha_before == gate_sha_after, "temporary_artifacts_present_during_smoke": temporary_artifacts_present_during_smoke, "temporary_artifacts_removed": not tmp.exists(), "protected_checkpoints_present": {k: v is not None for k, v in before.items()}, "protected_checkpoints_unchanged": protected_unchanged, "persistent_v7_training_started": False, "epoch2_training_authorized": False, "next_gate": "start_bounded_resumable_v7_first_epoch_training" if status == "pass" else "fix_v7_resume_before_persistent_training"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); args = parser.parse_args(); print(json.dumps(run_v7_resume_smoke(args.root), indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
