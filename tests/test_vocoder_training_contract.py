from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxMultiScaleWaveformDiscriminator,
    LykenoxVocoderConfig,
    LykenoxVocoderGenerator,
)
from lykenox_voice_engine.training.speech_vocoder_artifact import (
    load_vocoder_checkpoint,
    save_vocoder_checkpoint,
)
from lykenox_voice_engine.training.speech_vocoder_data import (
    VOCODER_SEGMENT_CONTRACT_VERSION,
    _stable_start,
)
from lykenox_voice_engine.training.speech_vocoder_losses import (
    VOCODER_LOSS_RECIPE_VERSION,
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_reconstruction_loss,
)


class VocoderTrainingContractTests(unittest.TestCase):
    def test_stable_segment_start_is_deterministic_and_bounded(self) -> None:
        first = _stable_start(
            seed=1337,
            split="train",
            utterance_id="u1",
            max_start=100,
            boundary_margin_frames=4,
        )
        second = _stable_start(
            seed=1337,
            split="train",
            utterance_id="u1",
            max_start=100,
            boundary_margin_frames=4,
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 4)
        self.assertLessEqual(first, 96)

    def test_perceptual_loss_stack_is_finite_and_differentiable(self) -> None:
        generator = LykenoxVocoderGenerator(
            LykenoxVocoderConfig(channels=32)
        )
        discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2)
        mel = torch.randn(1, 16, 80)
        target = torch.randn(1, 16 * 256).tanh()
        fake = generator(mel)
        reconstruction = multi_resolution_reconstruction_loss(fake, target)
        real_output = discriminator(target)
        fake_output = discriminator(fake)
        discriminator_loss = discriminator_hinge_loss(
            real_output,
            discriminator(fake.detach()),
        )
        generator_loss = (
            reconstruction.total
            + 0.1 * generator_adversarial_loss(fake_output)
            + 2.0 * feature_matching_loss(real_output, fake_output)
        )
        self.assertTrue(bool(torch.isfinite(discriminator_loss).item()))
        self.assertTrue(bool(torch.isfinite(generator_loss).item()))
        generator_loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in generator.parameters())
        )

    def test_checkpoint_roundtrip_restores_generator_and_discriminator(self) -> None:
        config = LykenoxVocoderConfig(channels=32)
        generator = LykenoxVocoderGenerator(config)
        discriminator = LykenoxMultiScaleWaveformDiscriminator(scales=2)
        provenance = {
            "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
            "loss_recipe_version": VOCODER_LOSS_RECIPE_VERSION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocoder.pt"
            save_vocoder_checkpoint(
                path,
                generator,
                discriminator,
                epoch=1,
                global_step=2,
                validation_reconstruction_loss=0.5,
                training_provenance=provenance,
            )
            restored_generator, restored_discriminator, payload = load_vocoder_checkpoint(path)
        self.assertEqual(restored_generator.config.to_dict(), config.to_dict())
        self.assertEqual(restored_discriminator.scales, discriminator.scales)
        self.assertEqual(
            payload["training_provenance"]["segment_contract_version"],
            VOCODER_SEGMENT_CONTRACT_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
