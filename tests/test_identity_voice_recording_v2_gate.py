from __future__ import annotations

from pathlib import Path

from lykenox_voice_engine.training import identity_voice_recording_v2 as gate


def test_recording_v2_gate_blocks_old_corpus_training_path() -> None:
    assert gate.POLICY_ID == "LYX-POL-001"
    assert gate.RECORDING_V2_VERSION == "lykenox-identity-voice-recording-v2"
    assert gate.OLD_132_CORPUS_PRIMARY_TRAINING_AUTHORIZED is False
    assert gate.AFFTDN_BATCH_AUTHORIZED is False
    assert gate.DEEPFILTERNET_132_BATCH_AUTHORIZED is False
    assert gate.RECORDING_V2_REQUIRED_BEFORE_NEW_PERSISTENT_TRAINING is True
    assert gate.RAW_RECORDING_V2_MUST_REMAIN_IMMUTABLE is True
    assert gate.EXTERNAL_EVENT_OVERLAP_REQUIRES_RETAKE_WHEN_PRACTICAL is True
    assert gate.HUMAN_AUDITORY_ACCEPTANCE_REQUIRED is True
    assert gate.ALL_DERIVED_TARGETS_MUST_BE_REGENERATED is True
    assert gate.GOLD_ORACLES_MUST_BE_RERUN is True
    assert gate.FURTHER_SOURCE_TRAINING_AUTHORIZED is False
    assert gate.FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED is False


def test_recording_v2_session_reuses_prompt_text_not_old_audio() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "prepare_identity_voice_recording_v2_session.py"
    text = script.read_text(encoding="utf-8")
    assert 'f"{old_id}_rec2"' in text
    assert '"old_audio_reused": False' in text
    assert '"old_text_and_split_assignment_reused": True' in text
    assert '"training_authorized": False' in text
    assert '"capture_status": "PENDING"' in text
    assert '"auditory_status": "PENDING"' in text


def test_recording_v2_decision_documents_retake_over_aggressive_cleanup() -> None:
    root = Path(__file__).resolve().parents[1]
    decision = root / "docs" / "LYKENOX_IDENTITY_VOICE_RECORDING_V2_DECISION_2026-09-04.md"
    text = decision.read_text(encoding="utf-8")
    assert "RECORDING_V2 clean recapture at source" in text
    assert "not authorized as the primary source for new persistent training" in text
    assert "retake" in text.lower()
    assert "Human listening remains the final authority" in text
