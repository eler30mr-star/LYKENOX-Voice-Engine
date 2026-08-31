from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from lykenox_voice_engine.models.speech.mel_postnet import (
    MEL_POSTNET_ARCHITECTURE_V1,
    LykenoxAcousticMelPostnetCandidate,
    MelResidualPostnetV1,
)
from lykenox_voice_engine.training.speech_acoustic_mel_fidelity_artifact import (
    ISOLATED_MEL_PERCEPTUALLY_REJECTED,
    ISOLATED_MEL_PERSISTENT_TRAINING_ENABLED,
    isolated_output_dir,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_smoke import (
    SMOKE_VERSION,
    run_mel_postnet_smoke,
)


class _DummyConfig:
    mel_bins = 8


class _DummyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyConfig()
        self.anchor = nn.Parameter(torch.tensor(1.0))

    def forward(self, token_ids, token_mask=None, durations=None):
        batch, frames = token_ids.shape[0], 5
        mel = torch.arange(batch * frames * 8, dtype=torch.float32).reshape(batch, frames, 8) / 100.0
        mask = torch.ones((batch, frames), dtype=torch.bool)
        return {
            "mel": mel,
            "mel_mask": mask,
            "mel_lengths": torch.full((batch,), frames, dtype=torch.long),
            "duration_prediction": torch.ones_like(token_ids, dtype=torch.float32),
            "regulated_durations": torch.ones_like(token_ids, dtype=torch.long),
            "f0_prediction_hz": torch.full((batch, frames), 100.0),
            "voicing_logits": torch.zeros((batch, frames)),
        }


class AcousticMelPostnetContractTests(unittest.TestCase):
    def test_zero_init_postnet_is_exact_identity(self) -> None:
        torch.manual_seed(7)
        postnet = MelResidualPostnetV1(8, hidden_channels=16)
        mel = torch.randn(2, 11, 8)
        mask = torch.ones(2, 11, dtype=torch.bool)
        self.assertTrue(torch.equal(postnet(mel, mask), mel))
        self.assertEqual(postnet.architecture, MEL_POSTNET_ARCHITECTURE_V1)

    def test_candidate_changes_only_mel_surface(self) -> None:
        base = _DummyBase()
        candidate = LykenoxAcousticMelPostnetCandidate(base, hidden_channels=16)
        token_ids = torch.ones((1, 3), dtype=torch.long)
        with torch.no_grad():
            base_output = base(token_ids)
            candidate_output = candidate(token_ids)
        self.assertTrue(torch.equal(candidate_output["mel"], base_output["mel"]))
        for key in (
            "duration_prediction",
            "regulated_durations",
            "f0_prediction_hz",
            "voicing_logits",
            "mel_mask",
            "mel_lengths",
        ):
            self.assertTrue(torch.equal(candidate_output[key], base_output[key]))
        self.assertTrue(all(name.startswith("postnet.") for name in candidate.trainable_parameter_names()))
        self.assertFalse(any(p.requires_grad for p in candidate.base_model.parameters()))

    def test_rejected_mel_decoder_persistent_path_is_blocked(self) -> None:
        self.assertTrue(ISOLATED_MEL_PERCEPTUALLY_REJECTED)
        self.assertFalse(ISOLATED_MEL_PERSISTENT_TRAINING_ENABLED)
        with self.assertRaises(RuntimeError):
            isolated_output_dir(torch.tensor(0))

    def test_product_smoke_is_nonpersistent_and_postnet_only(self) -> None:
        self.assertEqual(SMOKE_VERSION, "acoustic-mel-residual-postnet-product-smoke-v1")
        source = inspect.getsource(run_mel_postnet_smoke)
        self.assertIn("candidate.postnet.parameters()", source)
        self.assertNotIn("base.parameters()", source)
        self.assertNotIn("optimizer.step()\n        base", source)
        self.assertIn('"persistent_training_started": False', source)
        self.assertIn('"training_authorized": False', source)
        self.assertIn("frame_grid_artifact_metrics", source)
        self.assertIn("target_relative_presence_loss", source)


if __name__ == "__main__":
    unittest.main()
