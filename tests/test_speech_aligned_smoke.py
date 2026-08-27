from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.training.speech_aligned_smoke import validate_aligned_record


class SpeechAlignedSmokeTests(unittest.TestCase):
    def test_validate_aligned_record_returns_exact_teacher_durations(self) -> None:
        token_ids = torch.tensor([1, 5, 6, 2], dtype=torch.long)
        record = {
            "cache_version": "alignment-v3",
            "utterance_id": "utt-1",
            "text": "si",
            "token_ids": [1, 5, 6, 2],
            "durations": [3, 4, 5, 2],
        }
        durations = validate_aligned_record(
            record,
            utterance_id="utt-1",
            text="si",
            token_ids=token_ids,
            mel_frames=14,
        )
        self.assertEqual(durations.tolist(), [3, 4, 5, 2])

    def test_validate_aligned_record_rejects_mel_sum_mismatch(self) -> None:
        token_ids = torch.tensor([1, 5, 2], dtype=torch.long)
        record = {
            "cache_version": "alignment-v3",
            "utterance_id": "utt-2",
            "text": "a",
            "token_ids": [1, 5, 2],
            "durations": [2, 5, 2],
        }
        with self.assertRaises(RuntimeError):
            validate_aligned_record(
                record,
                utterance_id="utt-2",
                text="a",
                token_ids=token_ids,
                mel_frames=10,
            )

    def test_validate_aligned_record_rejects_stale_tokens(self) -> None:
        token_ids = torch.tensor([1, 5, 2], dtype=torch.long)
        record = {
            "cache_version": "alignment-v3",
            "utterance_id": "utt-3",
            "text": "a",
            "token_ids": [1, 6, 2],
            "durations": [2, 5, 2],
        }
        with self.assertRaises(RuntimeError):
            validate_aligned_record(
                record,
                utterance_id="utt-3",
                text="a",
                token_ids=token_ids,
                mel_frames=9,
            )


if __name__ == "__main__":
    unittest.main()
