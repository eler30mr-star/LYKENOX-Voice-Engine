from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.calibrate_identity_voice_clean_v1_ffmpeg_afftdn import (
    PROFILES,
    REQUIRED_AUDIT_IDS,
    select_trial_rows,
)
from scripts.run_identity_voice_clean_v1_ffmpeg_afftdn_batch import (
    run_clean_v1_ffmpeg_afftdn_batch,
)


def _write_wav(path: Path, *, level: float, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(sample_rate, level, dtype=np.float32)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def test_profiles_are_denoise_only_and_conservative() -> None:
    assert set(PROFILES) == {"conservative", "moderate"}
    assert "nr=6" in PROFILES["conservative"]
    assert "nr=10" in PROFILES["moderate"]
    for graph in PROFILES.values():
        assert graph.startswith("afftdn=")
        assert "loudnorm" not in graph
        assert "equalizer" not in graph
        assert "highpass" not in graph
        assert "lowpass" not in graph


def test_trial_selection_keeps_gold_ids_and_adds_noisiest(tmp_path: Path) -> None:
    root = tmp_path
    rows: list[dict[str, str]] = []
    ids = [*REQUIRED_AUDIT_IDS, "speech_extra_quiet", "speech_extra_noisy"]
    levels = [0.001, 0.002, 0.0002, 0.02]
    for utterance_id, level in zip(ids, levels, strict=True):
        wav = root / "source" / f"{utterance_id}.wav"
        _write_wav(wav, level=level)
        rows.append(
            {
                "utterance_id": utterance_id,
                "split": "val",
                "source_wav_path": str(wav),
                "source_sha256": "unused",
                "clean_wav_path": str(root / "clean" / f"{utterance_id}.wav"),
                "text": utterance_id,
            }
        )

    selected = select_trial_rows(root, rows, items=3)
    selected_ids = [row["utterance_id"] for row in selected]
    assert selected_ids[:2] == list(REQUIRED_AUDIT_IDS)
    assert selected_ids[2] == "speech_extra_noisy"


def test_batch_requires_explicit_valid_profile(tmp_path: Path) -> None:
    try:
        run_clean_v1_ffmpeg_afftdn_batch(tmp_path, profile="aggressive")
    except ValueError as error:
        assert "profile must be one of" in str(error)
    else:
        raise AssertionError("invalid profile unexpectedly accepted")
