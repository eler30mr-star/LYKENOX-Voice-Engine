from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch

from lykenox_voice_engine.training import speech_vocoder_owned_pipeline_forensics as audit
from lykenox_voice_engine.training.speech_pitch import PitchFrames
from lykenox_voice_engine.training.speech_vocoder_data import (
    OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
    VOCODER_SEGMENT_CONTRACT_VERSION,
    _slice_owned_segment,
    collect_owned_vocoder_segments,
)


class OwnedVocoderPipelineForensicsTests(unittest.TestCase):
    def test_v2_slices_all_conditioning_from_one_full_utterance_origin(self) -> None:
        frames = 10
        hop = 256
        mel = torch.arange(frames * 80, dtype=torch.float32).reshape(frames, 80)
        waveform = torch.arange(frames * hop, dtype=torch.float32)
        pitch = PitchFrames(
            f0_hz=torch.arange(frames, dtype=torch.float32) + 100.0,
            voiced=(torch.arange(frames) % 2).to(torch.float32),
            periodicity=torch.linspace(0.1, 0.9, frames),
        )
        segment = _slice_owned_segment(
            split="train",
            utterance_id="unit",
            wav_path=Path("unit.wav"),
            mel=mel,
            waveform=waveform,
            pitch=pitch,
            start_frame=2,
            segment_mel_frames=4,
            hop_length=hop,
        )
        self.assertEqual(
            OWNED_VOCODER_SEGMENT_CONTRACT_VERSION,
            "vocoder-segment-v2-full-utterance-mel-pitch-conditioning",
        )
        self.assertEqual(VOCODER_SEGMENT_CONTRACT_VERSION, "vocoder-segment-v1")
        self.assertTrue(torch.equal(segment.mel, mel[2:6]))
        self.assertTrue(torch.equal(segment.f0_hz, pitch.f0_hz[2:6]))
        self.assertTrue(torch.equal(segment.voiced, pitch.voiced[2:6]))
        self.assertTrue(torch.equal(segment.periodicity, pitch.periodicity[2:6]))
        self.assertTrue(torch.equal(segment.waveform, waveform[2 * hop : 6 * hop]))

    def test_v2_collector_uses_owned_pitch_cache_not_crop_reanalysis(self) -> None:
        source = inspect.getsource(collect_owned_vocoder_segments)
        self.assertIn("load_indexed_pitch_target", source)
        self.assertIn("_slice_owned_segment", source)
        self.assertNotIn("extract_pitch_frames", source)

    def test_forensics_is_read_only_and_blocks_model_work(self) -> None:
        source = inspect.getsource(audit)
        source_lower = source.lower()
        run_source = inspect.getsource(audit.run_owned_vocoder_conditioning_forensics)
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-vocoder-conditioning-pipeline-forensics-v1",
        )
        # Guard executable training mechanisms rather than incidental prose in docstrings.
        # The audit may accurately say "no optimizer step" without that being evidence of
        # optimizer creation or a parameter update.
        for forbidden in (
            "torch.optim.",
            "optim.adam(",
            "optim.adamw(",
            ".backward(",
            ".step(",
        ):
            self.assertNotIn(forbidden, source_lower)
        self.assertNotIn("torch.save(", source_lower)
        self.assertNotIn("from_pretrained", source_lower)
        self.assertIn('"training_started": False', run_source)
        self.assertIn('"persistent_training_authorized": False', run_source)
        self.assertIn('"new_vocoder_architecture_authorized": False', run_source)
        self.assertIn('"third_party_model_used": False', run_source)
        self.assertIn('"predicted_duration_modified": False', run_source)
        self.assertIn('"posthoc_gain_normalization_used": False', run_source)
        self.assertIn('"posthoc_eq_used": False', run_source)
        self.assertIn('"posthoc_denoising_used": False', run_source)

    def test_pitch_comparison_detects_crop_semantic_difference(self) -> None:
        old_f0 = torch.tensor([100.0, 0.0, 120.0, 130.0])
        new_f0 = torch.tensor([100.0, 110.0, 120.0, 130.0])
        old_voiced = torch.tensor([1.0, 0.0, 1.0, 1.0])
        new_voiced = torch.ones(4)
        old_periodicity = torch.tensor([0.5, 0.2, 0.6, 0.7])
        new_periodicity = torch.tensor([0.5, 0.4, 0.6, 0.7])
        result = audit._pitch_comparison(
            old_f0,
            old_voiced,
            old_periodicity,
            new_f0,
            new_voiced,
            new_periodicity,
        )
        self.assertGreater(float(result["voicing_disagreement_fraction"]), 0.0)
        self.assertGreater(float(result["periodicity_l1"]), 0.0)


if __name__ == "__main__":
    unittest.main()
