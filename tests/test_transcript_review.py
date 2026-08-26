"""Tests for applying transcript review corrections."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from lykenox_voice_engine.core.transcript_review import apply_transcripts


class TestTranscriptReview(unittest.TestCase):
    """Validate corrected transcript output generation."""

    def test_applies_corrected_text_without_overwriting_original_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_dir = Path(temp_dir)
            manifest_path = prepared_dir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "utterance_id": "speech_0001",
                        "wav_path": "take.wav",
                        "text": "Texto corto.",
                        "split": "train",
                        "needs_transcript_review": True,
                        "warning": "metadata incompleta",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_split_csv(prepared_dir / "train.csv", ["speech_0001"])
            _write_split_csv(prepared_dir / "val.csv", [])
            review_path = prepared_dir / "review.csv"
            with review_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["utterance_id", "corrected_text"])
                writer.writeheader()
                writer.writerow(
                    {
                        "utterance_id": "speech_0001",
                        "corrected_text": "Texto real leido por el usuario.",
                    }
                )

            result = apply_transcripts(prepared_dir, review_path, "unit-test", "auto")

            self.assertEqual(result.updated_count, 1)
            self.assertTrue(result.ready_for_trial_training)
            original = json.loads(manifest_path.read_text(encoding="utf-8").strip())
            updated = json.loads((prepared_dir / "manifest.auto.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(original["text"], "Texto corto.")
            self.assertEqual(updated["text"], "Texto real leido por el usuario.")
            self.assertEqual(updated["original_text"], "Texto corto.")
            self.assertEqual(updated["split"], "train")
            self.assertFalse(updated["needs_transcript_review"])

    def test_reports_missing_corrections_for_flagged_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_dir = Path(temp_dir)
            (prepared_dir / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "utterance_id": "speech_0001",
                        "wav_path": "take.wav",
                        "text": "Texto corto.",
                        "split": "val",
                        "needs_transcript_review": True,
                        "warning": "metadata incompleta",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _write_split_csv(prepared_dir / "train.csv", [])
            _write_split_csv(prepared_dir / "val.csv", ["speech_0001"])
            review_path = prepared_dir / "review.csv"
            with review_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["utterance_id", "corrected_text"])
                writer.writeheader()

            result = apply_transcripts(prepared_dir, review_path, "unit-test", "auto")

            self.assertEqual(result.missing_count, 1)
            self.assertFalse(result.ready_for_trial_training)


def _write_split_csv(path: Path, utterance_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["utterance_id", "wav_path", "text"])
        writer.writeheader()
        for utterance_id in utterance_ids:
            writer.writerow({"utterance_id": utterance_id, "wav_path": "take.wav", "text": "x"})


if __name__ == "__main__":
    unittest.main()
