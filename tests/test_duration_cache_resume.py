from __future__ import annotations

import unittest

from lykenox_voice_engine.training.speech_duration_cache import (
    DURATION_CACHE_VERSION,
    _record_is_reusable,
)


class DurationCacheResumeTests(unittest.TestCase):
    def _record(self) -> dict[str, object]:
        return {
            "cache_version": DURATION_CACHE_VERSION,
            "frontend_version": "es-phoneme-v1",
            "checkpoint_sha256": "abc123",
            "utterance_id": "utt-001",
            "text": "hola",
            "mel_frames": 100,
            "token_ids": [1, 7, 8, 2],
            "durations": [4, 40, 50, 6],
            "boundary_frames": {"leading": 4, "trailing": 6},
            "content": [
                {"position": 1, "token": "o", "token_id": 7, "duration_frames": 40},
                {"position": 2, "token": "l", "token_id": 8, "duration_frames": 50},
            ],
            "alignment_score_per_step": -0.5,
        }

    def test_matching_record_is_reusable(self) -> None:
        self.assertTrue(
            _record_is_reusable(
                self._record(),
                frontend_version="es-phoneme-v1",
                checkpoint_sha256_value="abc123",
                utterance_id="utt-001",
                text="hola",
                token_ids=[1, 7, 8, 2],
            )
        )

    def test_changed_checkpoint_or_text_invalidates_record(self) -> None:
        record = self._record()
        self.assertFalse(
            _record_is_reusable(
                record,
                frontend_version="es-phoneme-v1",
                checkpoint_sha256_value="different",
                utterance_id="utt-001",
                text="hola",
                token_ids=[1, 7, 8, 2],
            )
        )
        self.assertFalse(
            _record_is_reusable(
                record,
                frontend_version="es-phoneme-v1",
                checkpoint_sha256_value="abc123",
                utterance_id="utt-001",
                text="hola mundo",
                token_ids=[1, 7, 8, 2],
            )
        )


if __name__ == "__main__":
    unittest.main()
