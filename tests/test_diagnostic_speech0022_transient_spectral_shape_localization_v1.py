from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnostic_speech0022_transient_spectral_shape_localization_v1.py"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_transient_shape_gate_is_no_training_and_targeted() -> None:
    source = _source()
    assert 'TARGET_UTTERANCE_ID = "speech_0022_ba721f6129b9_seg_005"' in source
    assert "PREFIX_SECONDS = (0.5, 1.0, 1.5, 2.0)" in source
    assert "COMPLEMENT_SECONDS = 1.0" in source
    assert '"training_executed": False' in source
    assert '"optimizer_created": False' in source
    assert '"checkpoint_written": False' in source
    assert '"renderer_modified": False' in source
    assert '"posthoc_eq_used": False' in source
    assert '"posthoc_denoising_used": False' in source
    assert '"predicted_duration_modified": False' in source
    assert '"third_party_model_used": False' in source


def test_transient_shape_gate_freezes_target_phase_and_candidate_level() -> None:
    source = _source()
    assert "target_phase = torch.angle(target_spec)" in source
    assert "target_level, target_shape = _decompose_log_magnitude(target_mag)" in source
    assert "candidate_level, candidate_shape = _decompose_log_magnitude(candidate_mag)" in source
    assert "_compose_magnitude(candidate_level, shape)" in source
    assert "_compose_magnitude(candidate_level, target_shape)" in source
    assert "target_phase," in source
    assert "candidate_shape_first_1p0s_target_shape_after_render" in source


def test_transient_shape_gate_has_no_training_calls() -> None:
    tree = ast.parse(_source())
    forbidden_attributes = {"backward", "step", "zero_grad"}
    forbidden_names = {"Adam", "AdamW", "SGD"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in forbidden_attributes
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names


def test_transient_shape_gate_keeps_raw_and_common_audition_outputs() -> None:
    source = _source()
    assert 'raw_dir = output_dir / "raw"' in source
    assert 'audition_dir = output_dir / "audition"' in source
    assert "audition_gain = _common_audition_gain(renders, reference)" in source
    assert '"audition_monitor_gain_common_within_utterance": True' in source
    assert '"metrics_can_accept_product_quality": False' in source
