from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig
from lykenox_voice_engine.training.speech_acoustic_artifact import (
    load_speech_acoustic_checkpoint,
    save_speech_acoustic_checkpoint,
    vocabulary_sha256,
)
from lykenox_voice_engine.training.speech_aligned_data import collate_aligned_speech
from lykenox_voice_engine.training.speech_losses import masked_l1_loss


class SpeechTrainingContractTests(unittest.TestCase):
    def test_collate_preserves_lengths_and_masks_padding(self) -> None:
        items = [
            {
                "utterance_id": "a",
                "text": "a",
                "token_ids": torch.tensor([1, 2, 3]),
                "durations": torch.tensor([1, 2, 1]),
                "mel": torch.randn(4, 80),
            },
            {
                "utterance_id": "b",
                "text": "b",
                "token_ids": torch.tensor([1, 4]),
                "durations": torch.tensor([2, 1]),
                "mel": torch.randn(3, 80),
            },
        ]
        batch = collate_aligned_speech(items)
        self.assertEqual(tuple(batch.token_ids.shape), (2, 3))
        self.assertEqual(tuple(batch.mel.shape), (2, 4, 80))
        self.assertEqual(batch.token_lengths.tolist(), [3, 2])
        self.assertEqual(batch.mel_lengths.tolist(), [4, 3])
        self.assertFalse(bool(batch.token_mask[1, 2]))
        self.assertFalse(bool(batch.mel_mask[1, 3]))
        self.assertEqual(int(batch.durations[1, 2]), 0)
        self.assertTrue(torch.equal(batch.durations.sum(dim=1), batch.mel_lengths))

    def test_masked_mel_loss_ignores_padding_values(self) -> None:
        prediction = torch.zeros(2, 4, 3)
        target = torch.ones(2, 4, 3)
        mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]],
            dtype=torch.bool,
        )
        baseline = masked_l1_loss(prediction, target, mask)
        corrupted = target.clone()
        corrupted[1, 3] = 9999.0
        changed = masked_l1_loss(prediction, corrupted, mask)
        self.assertEqual(float(baseline), float(changed))

    def test_vectorized_regulator_preserves_per_item_lengths(self) -> None:
        encoded = torch.randn(2, 4, 5)
        durations = torch.tensor([[0, 2, 0, 3], [1, 0, 2, 0]])
        expanded, mask, lengths = LykenoxSpeechAcousticModel._length_regulate(
            encoded,
            durations,
        )
        self.assertEqual(lengths.tolist(), [5, 3])
        self.assertEqual(tuple(expanded.shape), (2, 5, 5))
        self.assertTrue(bool(mask[0].all()))
        self.assertTrue(bool(mask[1, :3].all()))
        self.assertFalse(bool(mask[1, 3:].any()))
        self.assertTrue(torch.equal(expanded[0, 0], encoded[0, 1]))
        self.assertTrue(torch.equal(expanded[0, 2], encoded[0, 3]))

    def test_padding_does_not_change_valid_duration_predictions(self) -> None:
        config = LykenoxSpeechConfig(
            vocab_size=32,
            hidden_size=32,
            encoder_layers=1,
            encoder_heads=4,
            dropout=0.0,
        )
        model = LykenoxSpeechAcousticModel(config).eval()
        single_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        single_mask = torch.ones_like(single_ids, dtype=torch.bool)
        batch_ids = torch.tensor([[5, 6, 7, 0], [5, 6, 7, 8]], dtype=torch.long)
        batch_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        )
        with torch.no_grad():
            single = model(single_ids, single_mask)["duration_prediction"][0]
            batched = model(batch_ids, batch_mask)["duration_prediction"][0, :3]
        self.assertTrue(torch.allclose(single, batched, atol=1e-6, rtol=1e-6))

    def test_checkpoint_roundtrip_binds_exact_vocabulary(self) -> None:
        frontend = SpanishTextFrontend()
        config = LykenoxSpeechConfig(
            vocab_size=frontend.vocab_size,
            hidden_size=32,
            encoder_layers=1,
            encoder_heads=4,
        )
        model = LykenoxSpeechAcousticModel(config)
        provenance = {
            "duration_cache_version": "alignment-v3",
            "train_manifest_sha256": "test",
            "val_manifest_sha256": "test",
            "duration_audit_sha256": "test",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "speech.pt"
            save_speech_acoustic_checkpoint(
                path,
                model,
                frontend=frontend,
                epoch=1,
                global_step=2,
                validation_loss=0.5,
                training_provenance=provenance,
            )
            restored, payload = load_speech_acoustic_checkpoint(path)
        self.assertEqual(restored.config.vocab_size, frontend.vocab_size)
        self.assertEqual(
            payload["vocabulary_sha256"],
            vocabulary_sha256(frontend.vocabulary()),
        )
        self.assertEqual(payload["frontend_version"], frontend.version)


if __name__ == "__main__":
    unittest.main()
