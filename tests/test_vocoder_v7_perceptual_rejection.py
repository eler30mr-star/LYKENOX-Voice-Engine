from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from lykenox_voice_engine.training.speech_vocoder_grid_artifact import (
    VOCODER_GRID_ARTIFACT_VERSION,
    frame_grid_artifact_metrics,
)
from lykenox_voice_engine.training.speech_vocoder_v7_rejection import (
    V7_PERCEPTUALLY_REJECTED,
    V7_TRAINING_ENABLED,
)
from lykenox_voice_engine.training.speech_vocoder_v7_train import (
    _run_config,
    run_bounded_resumable_v7_first_epoch,
)


class VocoderV7PerceptualRejectionTests(unittest.TestCase):
    def test_frame_grid_detector_rejects_hop_locked_tone(self) -> None:
        sample_rate = 24_000
        hop_length = 256
        samples = sample_rate * 2
        time = torch.arange(samples, dtype=torch.float32) / sample_rate
        frame_rate = sample_rate / hop_length
        waveform = (
            torch.sin(2.0 * torch.pi * frame_rate * time)
            + 0.8 * torch.sin(2.0 * torch.pi * frame_rate * 2.0 * time)
        )
        result = frame_grid_artifact_metrics(
            waveform,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        self.assertEqual(VOCODER_GRID_ARTIFACT_VERSION, "vocoder-frame-grid-artifact-v1")
        self.assertAlmostEqual(result.frame_rate_hz, 93.75)
        self.assertGreater(float(result.hop_autocorrelation[0]), 0.99)
        self.assertTrue(bool(result.severe_grid_artifact[0]))

    def test_frame_grid_detector_does_not_reject_seeded_noise(self) -> None:
        generator = torch.Generator().manual_seed(77)
        waveform = torch.randn(48_000, generator=generator)
        result = frame_grid_artifact_metrics(
            waveform,
            sample_rate=24_000,
            hop_length=256,
        )
        self.assertLess(abs(float(result.hop_autocorrelation[0])), 0.05)
        self.assertFalse(bool(result.severe_grid_artifact[0]))

    def test_v7_trainer_is_rejected_before_creating_artifacts(self) -> None:
        self.assertFalse(V7_TRAINING_ENABLED)
        self.assertTrue(V7_PERCEPTUALLY_REJECTED)
        config = _run_config(seed=77)
        self.assertFalse(config["training_enabled"])
        self.assertTrue(config["perceptually_rejected"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "frame-grid tones"):
                run_bounded_resumable_v7_first_epoch(root)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
