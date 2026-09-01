from __future__ import annotations

import inspect
import unittest

import torch
import torchaudio

from lykenox_voice_engine.models.speech import LykenoxSpeechConfig
from lykenox_voice_engine.training import speech_vocoder_loss_v2 as loss_v2


class OwnedVocoderLossV2Tests(unittest.TestCase):
    def test_valid_context_masks_remove_only_centered_crop_padding_frames(self) -> None:
        sample_count = 64 * 256
        expected = {
            (256, 64): (257, 253),
            (512, 128): (129, 125),
            (1024, 256): (65, 61),
        }
        for (n_fft, hop), (analysis_frames, valid_frames) in expected.items():
            mask = loss_v2.valid_centered_frame_mask(
                sample_count=sample_count,
                frame_count=analysis_frames,
                n_fft=n_fft,
                hop_length=hop,
            )
            self.assertEqual(int(mask.numel()), analysis_frames)
            self.assertEqual(int(mask.sum()), valid_frames)
            self.assertFalse(bool(mask[0]))
            self.assertFalse(bool(mask[-1]))

    def test_valid_context_stft_matches_full_utterance_on_every_scored_frame(self) -> None:
        torch.manual_seed(17)
        sample_count = 64 * 256
        start_sample = 8 * 256
        full = torch.randn(sample_count + 16 * 256, dtype=torch.float32)
        crop = full[start_sample : start_sample + sample_count]

        for n_fft, hop, win_length in loss_v2.STFT_RESOLUTIONS:
            local = loss_v2._centered_stft_magnitude(
                crop.unsqueeze(0),
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
            ).squeeze(0)
            whole = loss_v2._centered_stft_magnitude(
                full.unsqueeze(0),
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
            ).squeeze(0)
            start_frame = start_sample // hop
            whole_slice = whole[:, start_frame : start_frame + local.shape[1]]
            mask = loss_v2.valid_centered_frame_mask(
                sample_count=sample_count,
                frame_count=int(local.shape[1]),
                n_fft=n_fft,
                hop_length=hop,
            )
            self.assertTrue(
                torch.allclose(local[:, mask], whole_slice[:, mask], atol=1e-6, rtol=0.0)
            )

    def test_envelope_aligns_to_conditioning_and_ignores_unconditioned_terminal_frame(self) -> None:
        torch.manual_seed(23)
        config = LykenoxSpeechConfig()
        frames = 64
        sample_count = frames * config.hop_length
        start_sample = 8 * config.hop_length
        full = torch.randn(sample_count + 16 * config.hop_length, dtype=torch.float32)
        crop = full[start_sample : start_sample + sample_count]
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.mel_bins,
            power=1.0,
            center=True,
        )
        full_log_mel = torch.log(
            transform(full.unsqueeze(0)).clamp_min(1e-5)
        ).transpose(1, 2)
        conditioning = full_log_mel[
            :,
            start_sample // config.hop_length : start_sample // config.hop_length + frames,
            :,
        ]

        objective = loss_v2.ConditioningAlignedLogMelEnvelopeLossV2(config)
        result = objective(crop.unsqueeze(0), conditioning)
        self.assertEqual(result.conditioning_frames, 64)
        self.assertEqual(result.analysis_frames, 65)
        self.assertEqual(result.valid_conditioning_frames, 61)
        self.assertLess(float(result.log_mel_l1), 1e-6)
        self.assertLess(float(result.spectral_slope_l1), 1e-6)
        self.assertLess(float(result.temporal_delta_l1), 1e-6)

    def test_v2_objectives_are_finite_and_differentiable_without_model_or_persistence(self) -> None:
        torch.manual_seed(29)
        config = LykenoxSpeechConfig()
        frames = 64
        target = torch.randn(1, frames * config.hop_length).tanh()
        prediction = (target.detach() + 0.01 * torch.randn_like(target)).requires_grad_(True)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.mel_bins,
            power=1.0,
            center=True,
        )
        # Unit gradient test only: first T slots provide a shape-correct conditioning tensor.
        conditioning = torch.log(transform(target).clamp_min(1e-5)).transpose(1, 2)[:, :frames, :]
        reconstruction = loss_v2.valid_context_multi_resolution_reconstruction_loss(
            prediction,
            target,
        )
        envelope = loss_v2.ConditioningAlignedLogMelEnvelopeLossV2(config)(
            prediction,
            conditioning,
        )
        total = reconstruction.total + 0.5 * envelope.total
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_v2_contract_contains_no_architecture_checkpoint_or_external_model_route(self) -> None:
        source = inspect.getsource(loss_v2).lower()
        self.assertEqual(
            loss_v2.OWNED_VOCODER_LOSS_V2_VERSION,
            "owned-vocoder-loss-v2-valid-context-conditioning-aligned",
        )
        for forbidden in (
            "torch.optim.",
            "optim.adam(",
            "optim.adamw(",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "huggingface",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
