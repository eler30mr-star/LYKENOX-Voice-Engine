"""Tests for local LYKENOX readiness controls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.training_service import TrainingService


class TestTrainingService(unittest.TestCase):
    """Verify training controls stay aligned with the local voicebank route."""

    def test_check_reports_local_voicebank_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TrainingService(_minimal_root(Path(temp_dir)))

            result = service.check()

        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "lykenox_local_voicebank")
        self.assertEqual(result["neural_training"], "disabled")
        self.assertIn("voicebank", result)

    def test_neural_microtest_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = TrainingService(_minimal_root(Path(temp_dir)))

            result = service.microtest()

        self.assertFalse(result["ok"])
        self.assertIn("CPU", result["reason"])
        self.assertIn("multipitch", result["reason"])


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
    voicebank = root / "profiles" / "lykenox" / "voicebank"
    (voicebank / "wav").mkdir(parents=True)
    (voicebank / "reclist.txt").write_text("bai\nla\n", encoding="utf-8")
    (voicebank / "oto.ini").write_text("", encoding="utf-8")
    return root


if __name__ == "__main__":
    unittest.main()
