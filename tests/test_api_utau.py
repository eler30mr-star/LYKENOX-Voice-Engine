"""Tests for the UTAU sample backend API surface."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.api.schemas import NoteItem, SynthesizeRequest
from lykenox_voice_engine.api.server import create_app


class TestApiUtau(unittest.TestCase):
    """Validate API health and incomplete voicebank synthesis behavior."""

    def test_health_route_reports_utau_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _create_minimal_root(Path(temp_dir))
            app = create_app(root)
            endpoint = _endpoint(app, "/health")
            result = endpoint()
        self.assertEqual(result["backend"], "utau_sample")
        self.assertIn("voicebank_available", result)
        self.assertTrue(result["renderer_available"])

    def test_synthesize_without_voicebank_fails_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _create_minimal_root(Path(temp_dir))
            app = create_app(root)
            endpoint = _endpoint(app, "/synthesize")
            payload = SynthesizeRequest(
                profile="lykenox",
                lyrics="baila",
                tempo=120,
                notes=[NoteItem(lyric="bai", midi=60, start=0.0, duration=0.5)],
            )
            result = endpoint(payload)
        self.assertEqual(result.status, "failed")
        self.assertIn("Voicebank incompleto", result.error or "")

    def test_synthesize_with_incomplete_voicebank_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _create_minimal_root(Path(temp_dir))
            app = create_app(root)
            endpoint = _endpoint(app, "/synthesize")
            payload = SynthesizeRequest(
                profile="lykenox",
                lyrics="baila conmigo",
                tempo=120,
                notes=[NoteItem(lyric="bai", midi=60, start=0.0, duration=0.5)],
            )
            result = endpoint(payload)
        self.assertIn("bai", result.error or "")
        self.assertIsNone(result.output_path)


def _endpoint(app: object, path: str) -> object:
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"route missing: {path}")


def _create_minimal_root(root: Path) -> Path:
    config = root / "config"
    config.mkdir(parents=True)
    config.joinpath("app_settings.json").write_text(
        "{\"api_host\":\"127.0.0.1\",\"api_port\":8765,"
        "\"models_dir\":\"models\",\"profiles_dir\":\"profiles\","
        "\"datasets_dir\":\"datasets\",\"outputs_dir\":\"outputs\",\"device\":\"cpu\"}",
        encoding="utf-8",
    )
    profile = root / "profiles" / "lykenox"
    voicebank = profile / "voicebank"
    voicebank.mkdir(parents=True)
    (voicebank / "wav").mkdir()
    (voicebank / "reclist.txt").write_text("bai\nla\n", encoding="utf-8")
    (voicebank / "oto.ini").write_text("", encoding="utf-8")
    (profile / "profile.json").write_text('{"id":"lykenox","name":"LYKENOX"}', encoding="utf-8")
    (root / "datasets" / "lykenox" / "voicebank_raw").mkdir(parents=True)
    return root


if __name__ == "__main__":
    unittest.main()

