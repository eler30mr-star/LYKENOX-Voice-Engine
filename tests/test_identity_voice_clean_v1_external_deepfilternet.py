from __future__ import annotations

from pathlib import Path


def test_external_deepfilternet_calibration_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "calibrate_identity_voice_clean_v1_deepfilternet_external.py"
    text = path.read_text(encoding="utf-8")

    assert 'PROFILES: dict[str, float] = {' in text
    assert '"balanced": 24.0' in text
    assert '"full": 100.0' in text
    assert 'external_pretrained_model_used_for_offline_preparation' in text
    assert 'external_model_or_checkpoint_integrated_into_lykenox' in text
    assert 'lykenox_runtime_dependency_created' in text
    assert 'canonical_clean_v1_wav_written' in text
    assert '"pcm_f32le"' in text
    assert 'output_geometry_restored_to_source' in text
    assert 'human_auditory_quality_is_authority' in text

    # The external cleaner must stay an executable boundary, not an imported dependency.
    assert "import deepfilternet" not in text.lower()
    assert "from deepfilternet" not in text.lower()
    assert "--deepfilternet-exe" in text

    # Calibration must not silently authorize training or mutate CLEAN_V1 state.
    assert "training_authorized" not in text
    assert "clean_v1_state_path" not in text


def test_external_tool_setup_stays_outside_project_venv() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "setup_external_deepfilternet_rs.ps1"
    text = path.read_text(encoding="utf-8")
    assert "LYKENOX-external-tools" in text
    assert 'deepfilternet-rs-$Version' in text
    assert '"deepfilternet-rs"' in text
    assert '"0.1.1"' in text
    assert "lykenox_project_venv_modified=false" in text
    assert "-m pip install" in text
    assert ".venv\\Scripts\\python.exe" in text
    # The project venv may bootstrap creation, but package installation targets ToolPython only.
    assert '& $ToolPython -m pip install' in text


def test_afftdn_rejection_is_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "docs" / "LYKENOX_CLEAN_V1_AFFTDN_REJECTION_2026-09-04.md"
    text = path.read_text(encoding="utf-8")
    assert "AFFTDN" in text
    assert "REJECTED" in text
    assert "132-item" in text
    assert "must **not** be processed" in text
