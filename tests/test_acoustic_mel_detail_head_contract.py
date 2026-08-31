from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from lykenox_voice_engine.models.speech.mel_detail_head import (
    MEL_DETAIL_HEAD_ARCHITECTURE_V1,
    FrameHiddenMelDetailHeadV1,
    LykenoxAcousticFrameHiddenDetailCandidate,
)
from lykenox_voice_engine.training.speech_acoustic_mel_postnet_artifact import (
    MEL_POSTNET_PERCEPTUALLY_REJECTED,
    MEL_POSTNET_PERSISTENT_TRAINING_ENABLED,
    postnet_output_dir,
)
from lykenox_voice_engine.training.speech_acoustic_mel_detail_head_smoke import (
    SMOKE_VERSION,
    TRAIN_INDICES,
    VALIDATION_INDICES,
    run_mel_detail_head_smoke,
)


class _DummyConfig:
    hidden_size = 12
    mel_bins = 8


class _DummyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _DummyConfig()
        self.pre = nn.Linear(4, 12)
        self.mel_decoder = nn.Sequential(nn.Linear(12, 12), nn.GELU(), nn.Linear(12, 8))
        self.duration = nn.Linear(4, 1)

    def forward(self, token_ids, token_mask=None, durations=None):
        batch = token_ids.shape[0]
        frames = 5
        source = torch.arange(batch * frames * 4, dtype=torch.float32).reshape(batch, frames, 4) / 100.0
        frame_hidden = self.pre(source)
        mel = self.mel_decoder(frame_hidden)
        mask = torch.ones((batch, frames), dtype=torch.bool)
        duration_prediction = self.duration(source[:, : token_ids.shape[1]]).squeeze(-1)
        return {
            "mel": mel,
            "mel_mask": mask,
            "mel_lengths": torch.full((batch,), frames, dtype=torch.long),
            "duration_prediction": duration_prediction,
            "regulated_durations": torch.ones_like(token_ids, dtype=torch.long),
            "f0_prediction_hz": torch.full((batch, frames), 100.0),
            "voicing_logits": torch.zeros((batch, frames)),
        }


class AcousticMelDetailHeadContractTests(unittest.TestCase):
    def test_zero_init_head_outputs_zero_residual(self) -> None:
        torch.manual_seed(5)
        head = FrameHiddenMelDetailHeadV1(12, 8)
        hidden = torch.randn(2, 9, 12)
        mask = torch.ones((2, 9), dtype=torch.bool)
        residual = head(hidden, mask)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertEqual(head.architecture, MEL_DETAIL_HEAD_ARCHITECTURE_V1)

    def test_candidate_is_exact_identity_and_detail_head_only_trainable(self) -> None:
        base = _DummyBase()
        candidate = LykenoxAcousticFrameHiddenDetailCandidate(base)
        token_ids = torch.ones((1, 3), dtype=torch.long)
        with torch.no_grad():
            base_output = base(token_ids)
            output = candidate(token_ids)
        self.assertTrue(torch.equal(output["mel"], base_output["mel"]))
        self.assertTrue(torch.equal(output["base_mel"], base_output["mel"]))
        self.assertEqual(tuple(output["frame_hidden_for_detail"].shape), (1, 5, 12))
        for key in (
            "duration_prediction",
            "regulated_durations",
            "f0_prediction_hz",
            "voicing_logits",
            "mel_mask",
            "mel_lengths",
        ):
            self.assertTrue(torch.equal(output[key], base_output[key]))
        self.assertTrue(all(
            name.startswith("detail_head.") for name in candidate.trainable_parameter_names()
        ))
        self.assertFalse(any(parameter.requires_grad for parameter in candidate.base_model.parameters()))

    def test_candidate_captures_mel_decoder_input_not_output(self) -> None:
        source = inspect.getsource(LykenoxAcousticFrameHiddenDetailCandidate.forward)
        self.assertIn("register_forward_pre_hook", source)
        self.assertIn("captured.append(args[0].detach())", source)
        self.assertIn('output["frame_hidden_for_detail"]', source)
        self.assertNotIn('self.detail_head(base["mel"]', source)

    def test_rejected_postnet_persistent_training_is_blocked(self) -> None:
        self.assertTrue(MEL_POSTNET_PERCEPTUALLY_REJECTED)
        self.assertFalse(MEL_POSTNET_PERSISTENT_TRAINING_ENABLED)
        with self.assertRaises(RuntimeError):
            postnet_output_dir(torch.tensor(0))

    def test_smoke_uses_train_subset_and_three_heldout_product_gate(self) -> None:
        self.assertEqual(SMOKE_VERSION, "acoustic-frame-hidden-mel-detail-heldout-product-smoke-v1")
        self.assertEqual(TRAIN_INDICES, (0, 1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        source = inspect.getsource(run_mel_detail_head_smoke)
        self.assertIn("candidate.detail_head.parameters()", source)
        self.assertIn("presence_improved_items >= 2", source)
        self.assertIn("MAX_PRESENCE_REGRESSION_DB", source)
        self.assertIn("frame_grid_artifact_metrics", source)
        self.assertIn("target_relative_presence_loss", source)
        self.assertIn('"persistent_training_started": False', source)
        self.assertIn('"training_authorized": False', source)
        self.assertNotIn("torch.save", source)


if __name__ == "__main__":
    unittest.main()
