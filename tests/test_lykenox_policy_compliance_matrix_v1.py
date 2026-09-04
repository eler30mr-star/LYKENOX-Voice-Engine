from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "config" / "lykenox_policy_compliance_status_v1.json"
MATRIX_PATH = ROOT / "docs" / "LYKENOX_POLICY_COMPLIANCE_MATRIX.md"
ACTIVE_DECISION_PATH = (
    ROOT
    / "lykenox_voice_engine"
    / "training"
    / "speech_vocoder_active_source_decision.py"
)


def test_policy_compliance_status_contract() -> None:
    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert payload["policy_id"] == "LYX-POL-001"
    assert payload["overall_status"] == "COMPLIANT_DEVELOPMENT_WITH_PENDING_RELEASE_GATES"
    assert payload["matrix_document"] == "docs/LYKENOX_POLICY_COMPLIANCE_MATRIX.md"

    rules = payload["rules"]
    assert rules["R1_no_third_party_pretrained_weights"] == "PASS"
    assert rules["R2_no_third_party_neural_voice_components"] == "PASS"
    assert rules["R3_no_remote_inference_dependency"] == "PASS"
    assert rules["R6_no_probe_exception"] == "PASS"
    assert rules["R8_no_posthoc_quality_masking"] == "PASS"
    assert rules["R9_duration_contract_independent"] == "PASS"

    gate = payload["current_gate"]
    assert gate["name"] == "construct_and_audibly_validate_clean_v1_before_any_new_vocoder_training"
    assert gate["blocks_further_source_training"] is True
    assert gate["blocks_new_source_architecture"] is True

    assert all(payload["blocked_actions"].values())


def test_active_vocoder_decision_respects_current_policy_gate() -> None:
    text = ACTIVE_DECISION_PATH.read_text(encoding="utf-8")

    required_false_flags = (
        "FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False",
        "FURTHER_SOURCE_TRAINING_AUTHORIZED = False",
        "POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False",
        "POSTHOC_EQ_AUTHORIZED = False",
        "POSTHOC_DENOISING_AUTHORIZED = False",
        "PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False",
        "THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False",
        "REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False",
        "INFERENCE_TIME_TARGET_PHASE_COPY_AUTHORIZED = False",
    )
    for flag in required_false_flags:
        assert flag in text

    required_clean_v1_contract = (
        "CLEAN_V1_REQUIRED_BEFORE_FURTHER_SOURCE_TRAINING = True",
        "RAW_CORPUS_MUST_REMAIN_IMMUTABLE = True",
        "CLEAN_V1_ALL_ACOUSTIC_TARGETS_AND_CACHES_MUST_BE_REGENERATED = True",
        "DIRTY_WAV_DERIVED_TARGETS_MAY_BE_REUSED_AFTER_CLEAN_V1 = False",
        "VOCODER_EVALUATION_OUTPUT_DENOISE_AUTHORIZED = False",
    )
    for flag in required_clean_v1_contract:
        assert flag in text


def test_compliance_matrix_names_policy_and_current_gate() -> None:
    matrix = MATRIX_PATH.read_text(encoding="utf-8")

    assert "LYX-POL-001" in matrix
    assert "COMPLIANT_DEVELOPMENT_WITH_PENDING_RELEASE_GATES" in matrix
    assert "construct_and_audibly_validate_clean_v1_before_any_new_vocoder_training" in matrix
    assert "Denoiser externo preentrenado para limpiar dataset" in matrix
