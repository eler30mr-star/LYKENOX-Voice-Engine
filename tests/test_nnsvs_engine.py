"""Tests for the gated NNSVS backend integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.engines.nnsvs_engine import NnsvsEngine


class TestNnsvsEngine(unittest.TestCase):
    """Check that NNSVS is never treated as available silently."""

    def test_missing_backend_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = NnsvsEngine(Path(temp_dir))
            result = engine.check_available()
            self.assertFalse(result["available"])
            self.assertIn("nnsvs_env", result["reason"])

    def test_prepare_micro_score_without_raw_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "datasets" / "lykenox" / "raw").mkdir(parents=True)
            engine = NnsvsEngine(root)
            result = engine.prepare_dataset("lykenox")
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "needs_hts_full_context_labels")
            self.assertTrue(Path(result["score"]).exists())


if __name__ == "__main__":
    unittest.main()
