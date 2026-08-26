"""Tests for direct LYKENOX identity speech/singing API contracts."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from lykenox_voice_engine.api.server import create_app
from lykenox_voice_engine.api.schemas import SingRequest, SpeakRequest


class TestIdentityVoiceApi(unittest.TestCase):
    """Verify speech/singing target endpoints do not fake model readiness."""

    def test_health_declares_identity_target_and_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _minimal_root(Path(temp_dir))
            app = create_app(root)

            data = _endpoint(app, "/health")()

        self.assertEqual(data["backend"], "identity_voice_target")
        self.assertEqual(data["legacy_backend"], "utau_worldline_fallback")
        self.assertFalse(data["identity_model"]["uses_voice_conversion"])
        self.assertFalse(data["identity_model"]["uses_source_singer"])

    def test_speak_fails_honestly_until_identity_model_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _minimal_root(Path(temp_dir))
            app = create_app(root)

            response = _endpoint(app, "/speak")(
                SpeakRequest(profile="lykenox", text="Hola mundo", language="es")
            )

        data = response.model_dump()
        self.assertEqual(data["status"], "failed")
        self.assertIn("no entrenado", data["error"])
        self.assertIsNone(data["output_path"])

    def test_sing_fails_honestly_until_identity_model_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _minimal_root(Path(temp_dir))
            app = create_app(root)

            response = _endpoint(app, "/sing")(
                SingRequest(
                    profile="lykenox",
                    lyrics="baila conmigo",
                    tempo=120,
                    notes=[{"lyric": "bai", "midi": 60, "start": 0.0, "duration": 0.5}],
                )
            )

        data = response.model_dump()
        self.assertEqual(data["status"], "failed")
        self.assertIn("no entrenado", data["error"])
        self.assertIsNone(data["output_path"])


def _minimal_root(root: Path) -> Path:
    config = root / "config"
    config.mkdir()
    (config / "app_settings.json").write_text(
        json.dumps(
            {
                "api_host": "127.0.0.1",
                "api_port": 8765,
                "models_dir": "models",
                "profiles_dir": "profiles",
                "datasets_dir": "datasets",
                "outputs_dir": "outputs",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    profiles = root / "profiles" / "lykenox"
    profiles.mkdir(parents=True)
    (profiles / "profile.json").write_text(
        '{"id":"lykenox","name":"LYKENOX","language":"es","created_at":"test"}',
        encoding="utf-8",
    )
    voicebank = profiles / "voicebank"
    voicebank.mkdir()
    (voicebank / "reclist.txt").write_text("bai\n", encoding="utf-8")
    (voicebank / "oto.ini").write_text("", encoding="utf-8")
    (voicebank / "wav").mkdir()
    (root / "datasets" / "lykenox" / "voicebank_raw").mkdir(parents=True)
    return root


def _endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Endpoint not found: {path}")


if __name__ == "__main__":
    unittest.main()
