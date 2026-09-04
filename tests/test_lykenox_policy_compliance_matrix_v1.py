from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "LYKENOX_IDENTITY_DISTRIBUTION_POLICY.md"
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
    assert payload["policy_version"] == "1.1"
    assert payload["matrix_version"] == "1.1"
    assert payload["overall_status"] == "COMPLIANT_DEVELOPMENT_WITH_PENDING_RELEASE_GATES"
    assert payload["matrix_document"] == "docs/LYKENOX_POLICY_COMPLIANCE_MATRIX.md"

    boundary = payload["product_boundary"]
    assert boundary["third_party_model_implementation_in_lykenox_allowed"] is False
    assert boundary["third_party_pretrained_weights_in_lykenox_allowed"] is False
    assert boundary["remote_service_required_for_lykenox_inference_allowed"] is False
    assert boundary["external_offline_tools_outside_project_allowed"] is True
    assert boundary["external_offline_learned_tools_outside_project_allowed"] is True
    assert boundary["external_tool_outputs_may_enter_as_authorized_data"] is True
    assert boundary["external_tool_weights_may_enter_project"] is False
    assert boundary["external_tool_may_initialize_or_distill_lykenox_weights"] is False

    rules = payload["rules"]
    assert rules["R1_no_third_party_pretrained_weights_inside_lykenox"] == "PASS"
    assert rules["R2_no_third_party_neural_voice_components_in_product"] == "PASS"
    assert rules["R3_no_remote_inference_dependency"] == "PASS"
    assert rules["R6_external_offline_tools_allowed_without_integration"] == "PASS_POLICY_PENDING_PER_TOOL_RIGHTS_REVIEW"
    assert rules["R8_no_posthoc_quality_masking"] == "PASS"
    assert rules["R9_duration_contract_independent"] == "PASS"

    gate = payload["current_gate"]
    assert gate["name"] == "construct_and_audibly_validate_clean_v1_before_any_new_vocoder_training"
    assert gate["blocks_further_source_training"] is True
    assert gate["blocks_new_source_architecture"] is True

    assert all(payload["blocked_actions"].values())
    assert all(payload["permitted_external_offline_actions"].values())
    assert all(payload["external_offline_tool_conditions"].values())


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


def test_policy_v11_locks_zero_integration_but_allows_external_offline_tools() -> None:
    policy = POLICY_PATH.read_text(encoding="utf-8")

    assert "**Versión:** 1.1" in policy
    assert "Herramienta externa offline" in policy
    assert "cero implementación" in policy.lower() or "implementación" in policy.lower()
    assert "no se incluya ni se invoque desde el repositorio" in policy
    assert "no se use para inicializar, destilar, transferir o derivar pesos LYKENOX" in policy
    assert "una herramienta externa de limpieza puede utilizarse aunque sea aprendida" in policy
    assert "CLEAN_V1" in policy


def test_compliance_matrix_names_policy_current_gate_and_tool_boundary() -> None:
    matrix = MATRIX_PATH.read_text(encoding="utf-8")

    assert "LYX-POL-001" in matrix
    assert "Versión de política:** 1.1" in matrix
    assert "COMPLIANT_DEVELOPMENT_WITH_PENDING_RELEASE_GATES" in matrix
    assert "construct_and_audibly_validate_clean_v1_before_any_new_vocoder_training" in matrix
    assert "Implementar denoiser/separador externo dentro de LYKENOX" in matrix
    assert "Usar herramienta externa offline para limpiar datos" in matrix
    assert "PERMITTED" in matrix
