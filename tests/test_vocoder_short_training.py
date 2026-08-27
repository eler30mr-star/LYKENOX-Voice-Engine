from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.training.speech_vocoder_data import VocoderSegment
from lykenox_voice_engine.training.speech_vocoder_short_train import (
    SHORT_TRAIN_CONTRACT_VERSION,
    _segment_set_sha256,
)


class VocoderShortTrainingTests(unittest.TestCase):
    def _segment(self, utterance_id: str, start_frame: int) -> VocoderSegment:
        return VocoderSegment(
            split="train",
            utterance_id=utterance_id,
            wav_path=f"{utterance_id}.wav",
            start_frame=start_frame,
            mel_frames=96,
            mel=torch.zeros(96, 80),
            waveform=torch.zeros(96 * 256),
        )

    def test_short_training_contract_is_versioned(self) -> None:
        self.assertEqual(SHORT_TRAIN_CONTRACT_VERSION, "vocoder-short-train-v1")

    def test_segment_set_checksum_is_deterministic_and_order_sensitive(self) -> None:
        first = self._segment("a", 10)
        second = self._segment("b", 20)
        checksum = _segment_set_sha256([first, second])
        self.assertEqual(checksum, _segment_set_sha256([first, second]))
        self.assertNotEqual(checksum, _segment_set_sha256([second, first]))
        self.assertNotEqual(checksum, _segment_set_sha256([self._segment("a", 11), second]))


if __name__ == "__main__":
    unittest.main()
