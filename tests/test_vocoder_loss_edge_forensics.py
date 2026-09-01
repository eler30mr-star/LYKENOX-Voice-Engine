from __future__ import annotations

import inspect
import unittest

import torch
import torchaudio

from lykenox_voice_engine.training import speech_vocoder_loss_edge_forensics as audit


class VocoderLossEdgeForensicsTests(unittest.TestCase):
    def test_centered_crop_analysis_differs_only_where_crop_context_is_artificial(self) -> None:
        torch.manual_seed(7)
        n_fft = 1024
        hop = 256
        crop_samples = 64 * hop
        start_sample = 8 * hop
        full = torch.randn(crop_samples + 16 * hop, dtype=torch.float32)
        crop = full[start_sample : start_sample + crop_samples]

        local = audit._centered_stft_magnitude(
            crop,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
        )
        whole = audit._centered_stft_magnitude(
            full,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
        )
        start_frame = start_sample // hop
        whole_slice = whole[:, start_frame : start_frame + local.shape[1]]
        mask = audit._artificial_context_mask(
            sample_count=crop_samples,
            frame_count=int(local.shape[1]),
            n_fft=n_fft,
            hop_length=hop,
        )
        summary = audit._frame_error_summary(local, whole_slice, mask)

        self.assertEqual(int(local.shape[1]), 65)
        self.assertEqual(int(summary["artificial_context_frames"]), 4)
        self.assertGreater(float(summary["artificial_log_magnitude_l1"]), 1e-4)
        self.assertLess(float(summary["interior_log_magnitude_l1"]), 1e-6)

    def test_centered_crop_mel_has_one_more_frame_than_conditioning(self) -> None:
        torch.manual_seed(11)
        n_fft = 1024
        hop = 256
        frames = 64
        crop_samples = frames * hop
        start_sample = 8 * hop
        full = torch.randn(crop_samples + 16 * hop, dtype=torch.float32)
        crop = full[start_sample : start_sample + crop_samples]
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=80,
            power=1.0,
            center=True,
        )
        full_log_mel = torch.log(transform(full.unsqueeze(0)).clamp_min(1e-5)).squeeze(0).transpose(0, 1)
        crop_log_mel = torch.log(transform(crop.unsqueeze(0)).clamp_min(1e-5)).squeeze(0).transpose(0, 1)
        conditioning = full_log_mel[start_sample // hop : start_sample // hop + frames]
        summary = audit._mel_crop_vs_cached_summary(
            crop_log_mel,
            conditioning,
            n_fft=n_fft,
            hop_length=hop,
            sample_count=crop_samples,
        )

        self.assertEqual(summary["conditioning_frames"], 64)
        self.assertEqual(summary["crop_local_frames"], 65)
        self.assertEqual(summary["extra_crop_local_frames_without_conditioning"], 1)
        self.assertTrue(summary["has_unconditioned_terminal_analysis_frame"])
        self.assertGreater(float(summary["artificial_log_mel_l1"]), 1e-5)
        self.assertLess(float(summary["interior_log_mel_l1"]), 1e-6)

    def test_forensics_is_data_only_and_cannot_train_or_persist_models(self) -> None:
        source = inspect.getsource(audit).lower()
        run_source = inspect.getsource(audit.run_owned_vocoder_loss_edge_forensics)
        self.assertEqual(
            audit.AUDIT_VERSION,
            "owned-vocoder-loss-edge-semantics-forensics-v1",
        )
        for forbidden in (
            "torch.optim.",
            "optim.adam(",
            "optim.adamw(",
            ".backward(",
            ".step(",
            "torch.save(",
            "from_pretrained",
        ):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("lykenoxvocodergenerator", source)
        self.assertIn('"training_started": False', run_source)
        self.assertIn('"persistent_training_authorized": False', run_source)
        self.assertIn('"new_vocoder_architecture_authorized": False', run_source)
        self.assertIn('"third_party_model_used": False', run_source)


if __name__ == "__main__":
    unittest.main()
