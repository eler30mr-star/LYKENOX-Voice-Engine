from __future__ import annotations

import inspect
import math
import unittest

import torch

from lykenox_voice_engine.training.speech_vocoder_v4_2_reference_waveform_forensics import (
    AUDIT_VERSION,
    VALIDATION_INDICES,
    _frames,
    _pitch_metrics,
    run_v4_2_reference_waveform_forensics,
)


class V42ReferenceWaveformForensicsTests(unittest.TestCase):
    def test_contract_is_direct_oracle_and_read_only(self) -> None:
        self.assertEqual(AUDIT_VERSION, "vocoder-v4-2-reference-waveform-forensics-v1")
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        source = inspect.getsource(run_v4_2_reference_waveform_forensics)
        self.assertIn("target_mel = batch.mel", source)
        self.assertIn("target_f0 = batch.f0_hz", source)
        self.assertIn("target_voiced = batch.voiced", source)
        self.assertIn("vocoder(target_mel, target_f0, target_voiced)", source)
        self.assertIn('"reference_vs_oracle_direct_comparison": True', source)
        self.assertIn('"acoustic_training_authorized": False', source)
        self.assertIn('"vocoder_training_authorized": False', source)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            '"posthoc_gain_normalization_used": True',
            '"posthoc_eq_used": True',
            '"posthoc_denoising_used": True',
            '"predicted_duration_modified": True',
        ):
            self.assertNotIn(forbidden, source)

    def test_pitch_periodicity_tracks_known_sine(self) -> None:
        sample_rate = 24000
        hop = 256
        frame_count = 24
        frequency = 120.0
        samples = frame_count * hop
        time = torch.arange(samples, dtype=torch.float32) / sample_rate
        wave = 0.2 * torch.sin(2.0 * math.pi * frequency * time)
        frames = _frames(wave, frame_count, hop)
        target_f0 = torch.full((frame_count,), frequency)
        voiced = torch.ones((frame_count,), dtype=torch.bool)
        metrics = _pitch_metrics(frames, target_f0, voiced, sample_rate)
        self.assertGreater(metrics["voiced_periodicity"], 0.5)
        self.assertLess(metrics["voiced_pitch_mae_cents"], 80.0)
        self.assertEqual(metrics["unvoiced_periodicity"], 0.0)


if __name__ == "__main__":
    unittest.main()
