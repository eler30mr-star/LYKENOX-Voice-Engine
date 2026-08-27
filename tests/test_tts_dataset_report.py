"""Tests for neural TTS dataset readiness reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.tts_dataset_report import build_tts_dataset_report


class TestTtsDatasetReport(unittest.TestCase):
    """Verify dataset readiness is reported without implying a trained model."""

    def test_report_uses_segmented_train_val_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "datasets" / "lykenox" / "identity_voice" / "prepared"
            speech = base / "speech"
            segmented = base / "speech_segmented"
            speech.mkdir(parents=True)
            segmented.mkdir(parents=True)
            (speech / "quality_report.filtered.json").write_text(
                json.dumps({"usable_minutes": 1.0, "blocked_rows": 2, "blocked_reasons": {"x": 2}}),
                encoding="utf-8",
            )
            (segmented / "quality_report.segmented.json").write_text(
                json.dumps({"minutes": 3.5, "segments": 4, "ready_for_trial_training": True}),
                encoding="utf-8",
            )
            (segmented / "train.segmented.csv").write_text("utterance_id,wav_path,text\na,b,c\n", encoding="utf-8")
            (segmented / "val.segmented.csv").write_text("utterance_id,wav_path,text\nd,e,f\n", encoding="utf-8")

            report = build_tts_dataset_report(root)

        self.assertEqual(report["status"], "dataset_prepared_not_trained")
        self.assertFalse(report["model_trained"])
        self.assertEqual(report["train_rows"], 1)
        self.assertEqual(report["val_rows"], 1)
        self.assertTrue(report["ready_for_backend_microtest"])


if __name__ == "__main__":
    unittest.main()
