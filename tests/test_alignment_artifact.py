from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxCTCAligner, LykenoxCTCAlignerConfig
from lykenox_voice_engine.training.alignment_artifact import (
    checkpoint_sha256,
    load_aligner_checkpoint,
    save_aligner_checkpoint,
)


class LykenoxAlignmentArtifactTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_versioned_and_frontend_bound(self) -> None:
        frontend = SpanishTextFrontend()
        config = LykenoxCTCAlignerConfig(
            num_symbols=frontend.vocab_size,
            hidden_size=32,
            recurrent_layers=1,
        )
        model = LykenoxCTCAligner(config)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            save_aligner_checkpoint(
                path,
                model,
                frontend=frontend,
                speech_config={"sample_rate": 24000, "mel_bins": 80},
                epoch=3,
                validation_ctc_loss=1.25,
                training_metadata={"seed": 1337},
            )
            loaded, payload = load_aligner_checkpoint(path)
            self.assertEqual(payload["frontend_version"], frontend.version)
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(loaded.config.num_symbols, frontend.vocab_size)
            self.assertEqual(len(checkpoint_sha256(path)), 64)

            original = model.state_dict()
            restored = loaded.state_dict()
            self.assertEqual(original.keys(), restored.keys())
            for key in original:
                self.assertTrue(torch.equal(original[key], restored[key]))


if __name__ == "__main__":
    unittest.main()
