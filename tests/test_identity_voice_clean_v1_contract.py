from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lykenox_voice_engine.training.identity_voice_clean_v1 import (
    clean_v1_is_active,
    clean_v1_manifest_path,
    clean_v1_review_path,
    clean_v1_state_path,
    require_clean_v1_training_ready,
    resolve_identity_speech_manifest,
    sha256_file,
)
from scripts.activate_identity_voice_clean_v1 import activate_clean_v1
from scripts.prepare_identity_voice_clean_v1 import prepare_clean_v1
from scripts.validate_identity_voice_clean_v1 import validate_clean_v1


def _write_wave(path: Path, *, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24000
    samples = np.arange(sample_rate, dtype=np.float32)
    waveform = (0.08 * np.sin(2.0 * np.pi * frequency * samples / sample_rate)).astype(np.float32)
    sf.write(str(path), waveform, sample_rate, subtype="FLOAT")


def _write_source_split(root: Path, split: str, utterance_id: str, frequency: float) -> Path:
    base = root / "datasets" / "lykenox" / "identity_voice" / "prepared" / "speech_segmented"
    wav = base / "wav" / f"{utterance_id}.wav"
    _write_wave(wav, frequency=frequency)
    manifest = base / f"{split}.segmented.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utterance_id", "wav_path", "text"])
        writer.writeheader()
        writer.writerow({"utterance_id": utterance_id, "wav_path": str(wav), "text": "Prueba de voz."})
    return wav


def _accept_all_review(root: Path) -> None:
    path = clean_v1_review_path(root)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["technical_status"] == "PASS":
            row["auditory_decision"] = "ACCEPT"
            row["auditory_notes"] = "human test approval"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_clean_v1_prepare_validate_activate_and_training_gate(tmp_path: Path) -> None:
    train_source = _write_source_split(tmp_path, "train", "speech_train_seg_001", 180.0)
    val_source = _write_source_split(tmp_path, "val", "speech_val_seg_001", 220.0)
    train_hash = sha256_file(train_source)
    val_hash = sha256_file(val_source)

    prepared = prepare_clean_v1(tmp_path)
    assert prepared["status"] == "ready_for_external_cleaning"
    assert prepared["source_audio_mutated"] is False
    assert clean_v1_is_active(tmp_path) is False

    clean_wav_dir = tmp_path / "datasets" / "lykenox" / "identity_voice" / "clean_v1" / "wav"
    _write_wave(clean_wav_dir / "speech_train_seg_001.wav", frequency=180.0)
    _write_wave(clean_wav_dir / "speech_val_seg_001.wav", frequency=220.0)

    validation = validate_clean_v1(
        tmp_path,
        tool_name="offline-test-cleaner",
        tool_version="1.0",
        tool_terms_note="test fixture output authorized for test use",
    )
    assert validation["status"] == "ready_for_auditory_review"
    assert validation["technical_validation_passed"] is True
    assert validation["metrics_can_accept_perceptual_quality"] is False
    assert sha256_file(train_source) == train_hash
    assert sha256_file(val_source) == val_hash

    with pytest.raises(RuntimeError, match="technical validation and human auditory approval"):
        # Before explicit human approval CLEAN_V1 cannot be active.
        from lykenox_voice_engine.training.identity_voice_clean_v1 import require_clean_v1_active

        require_clean_v1_active(tmp_path, purpose="test training")

    _accept_all_review(tmp_path)
    activation = activate_clean_v1(tmp_path)
    assert activation["status"] == "active"
    assert activation["human_auditory_review_complete"] is True
    assert activation["training_authorized"] is False
    assert clean_v1_is_active(tmp_path) is True

    for split in ("train", "val"):
        manifest = clean_v1_manifest_path(tmp_path, split)
        assert manifest.exists()
        assert resolve_identity_speech_manifest(tmp_path, split) == manifest
        text = manifest.read_text(encoding="utf-8")
        assert "../wav/" in text

    with pytest.raises(RuntimeError, match="pending gates"):
        require_clean_v1_training_ready(tmp_path, purpose="source training")

    state = json.loads(clean_v1_state_path(tmp_path).read_text(encoding="utf-8"))
    assert state["all_acoustic_targets_and_caches_regenerated"] is False
    assert state["gold_oracles_rerun_after_clean_v1"] is False
    assert state["external_tool_integrated_into_lykenox"] is False
    assert sha256_file(train_source) == train_hash
    assert sha256_file(val_source) == val_hash
