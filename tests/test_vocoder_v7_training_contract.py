from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from lykenox_voice_engine.models.vocoder import LykenoxVocoderGeneratorV7, VOCODER_GENERATOR_V7_ARCHITECTURE
from lykenox_voice_engine.training.speech_pitch import PITCH_TARGET_VERSION
from lykenox_voice_engine.training.speech_vocoder_data import VOCODER_SEGMENT_CONTRACT_VERSION
from lykenox_voice_engine.training.speech_vocoder_v7_artifact import V7_TRAINING_PHASE, load_v7_checkpoint, save_v7_checkpoint
from lykenox_voice_engine.training.speech_vocoder_v7_content_loss import VOCODER_V7_CONTENT_LOSS_VERSION
from lykenox_voice_engine.training.speech_vocoder_v7_train import ARTIFACT_DIR_NAME, TRAINER_CONTRACT_VERSION, _run_config


class VocoderV7TrainingContractTests(unittest.TestCase):
    def _provenance(self) -> dict[str, object]:
        return {
            "generator_architecture": VOCODER_GENERATOR_V7_ARCHITECTURE,
            "source_family": LykenoxVocoderGeneratorV7.source_family,
            "source_free": True,
            "sample_phase_conditioning": False,
            "sample_rate_pitch_features": False,
            "pitch_conditioning_scope": "frame_latent_only",
            "deterministic_noise_conditioning": False,
            "local_unit_rms_shape_normalization": False,
            "global_unit_rms_shape_normalization": False,
            "level_rescue_branch": False,
            "training_phase": V7_TRAINING_PHASE,
            "segment_contract_version": VOCODER_SEGMENT_CONTRACT_VERSION,
            "pitch_target_version": PITCH_TARGET_VERSION,
            "v7_content_loss_version": VOCODER_V7_CONTENT_LOSS_VERSION,
        }

    def test_run_config_hard_blocks_second_epoch_contract(self) -> None:
        config = _run_config(seed=77000)
        self.assertEqual(config["trainer_contract_version"], TRAINER_CONTRACT_VERSION)
        self.assertEqual(config["hard_epoch_limit"], 1)
        self.assertTrue(config["source_free"])
        self.assertFalse(config["sample_phase_conditioning"])
        self.assertFalse(config["sample_rate_pitch_features"])
        self.assertFalse(config["deterministic_noise_conditioning"])
        self.assertFalse(config["level_rescue_branch"])
        self.assertEqual(config["v7_content_loss_version"], VOCODER_V7_CONTENT_LOSS_VERSION)
        self.assertNotIn("v6", ARTIFACT_DIR_NAME)

    def test_checkpoint_roundtrip_preserves_source_free_contract_and_optimizer(self) -> None:
        torch.manual_seed(7)
        model = LykenoxVocoderGeneratorV7(frame_channels=96, upsample_channels=(80, 56, 40), residual_kernels=(3, 7), residual_dilations=(1, 3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            save_v7_checkpoint(path, model, epoch=1, global_step=3, next_item_offset=3, validation_metrics={"selection_score": 1.0}, training_provenance=self._provenance(), generator_optimizer=optimizer, training_metadata={"run_config": _run_config(seed=77000)})
            loaded, payload = load_v7_checkpoint(path)
        self.assertTrue(payload["source_free"])
        self.assertFalse(payload["sample_phase_conditioning"])
        self.assertFalse(payload["sample_rate_pitch_features"])
        self.assertFalse(payload["deterministic_noise_conditioning"])
        self.assertFalse(payload["level_rescue_branch"])
        self.assertIsInstance(payload["generator_optimizer_state"], dict)
        self.assertEqual(payload["global_step"], 3)
        self.assertEqual(payload["next_item_offset"], 3)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]))

    def test_checkpoint_loader_rejects_false_source_free_claim(self) -> None:
        torch.manual_seed(11)
        model = LykenoxVocoderGeneratorV7(frame_channels=96, upsample_channels=(80, 56, 40), residual_kernels=(3, 7), residual_dilations=(1, 3))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.pt"
            save_v7_checkpoint(path, model, epoch=1, global_step=0, next_item_offset=0, validation_metrics=None, training_provenance=self._provenance())
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["sample_phase_conditioning"] = True
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "sample_phase_conditioning"):
                load_v7_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
