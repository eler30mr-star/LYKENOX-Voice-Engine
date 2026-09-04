from __future__ import annotations

from pathlib import Path


def test_lykenox_desktop_does_not_embed_recording_v2_capture() -> None:
    root = Path(__file__).resolve().parents[1]
    main = (root / "lykenox_voice_engine" / "ui" / "main_window.py").read_text(encoding="utf-8")

    assert "RecordingV2Page" not in main
    assert "recording_v2_bootstrap" not in main
    assert '"Grabar RECORDING_V2"' not in main
    assert "RecVoice" in main
    assert not (root / "lykenox_voice_engine" / "ui" / "recording_v2_page.py").exists()
    assert not (root / "lykenox_voice_engine" / "ui" / "recording_v2_bootstrap.py").exists()


def test_recording_v2_dataset_contract_remains_in_lykenox() -> None:
    root = Path(__file__).resolve().parents[1]

    # Moving capture UI out must not remove the authoritative dataset/training gate.
    assert (root / "lykenox_voice_engine" / "training" / "identity_voice_recording_v2.py").is_file()
    assert (root / "scripts" / "prepare_identity_voice_recording_v2_session.py").is_file()
    assert (root / "scripts" / "prepare_identity_voice_recording_v2_pilot.py").is_file()
    assert (root / "scripts" / "validate_identity_voice_recording_v2_pilot.py").is_file()
