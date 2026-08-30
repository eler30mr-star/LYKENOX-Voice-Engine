from __future__ import annotations

import unittest

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV6
from lykenox_voice_engine.training.speech_vocoder_v6_train import (
    TRAINER_CONTRACT_VERSION,
    _optimizer,
)


class VocoderV6TrainingContractTests(unittest.TestCase):
    def test_level_parameters_have_dedicated_optimizer_group(self) -> None:
        model = LykenoxVocoderGeneratorV6(
            frame_channels=64,
            upsample_channels=(48, 40, 32, 24),
            sample_channels=32,
            sample_dilations=(1, 2, 4, 8),
        )
        base_lr = 2e-4
        multiplier = 4.0
        optimizer = _optimizer(model, base_lr, multiplier)

        self.assertEqual(TRAINER_CONTRACT_VERSION, "v6-bounded-resumable-v1")
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], base_lr)
        self.assertEqual(optimizer.param_groups[1]["lr"], base_lr * multiplier)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

        shape_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        level_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
        expected_level_ids = {id(parameter) for parameter in model.level_parameters()}
        all_model_ids = {id(parameter) for parameter in model.parameters()}

        self.assertTrue(shape_ids)
        self.assertTrue(level_ids)
        self.assertTrue(shape_ids.isdisjoint(level_ids))
        self.assertEqual(level_ids, expected_level_ids)
        self.assertEqual(shape_ids | level_ids, all_model_ids)


if __name__ == "__main__":
    unittest.main()
