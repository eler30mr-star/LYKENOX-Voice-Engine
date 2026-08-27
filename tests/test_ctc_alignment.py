from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from lykenox_voice_engine.core.ctc_alignment import (
    ctc_targets,
    expand_content_durations,
    forced_alignment_durations,
)
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend
from lykenox_voice_engine.models.speech import LykenoxCTCAligner, LykenoxCTCAlignerConfig


class LykenoxCTCAlignmentTests(unittest.TestCase):
    def test_ctc_targets_strip_only_structural_tokens(self) -> None:
        frontend = SpanishTextFrontend()
        encoded = torch.tensor(frontend.encode("hola mundo"), dtype=torch.long)
        targets, positions = ctc_targets(encoded)
        self.assertEqual(len(positions), targets.numel())
        self.assertEqual(positions[0], 1)
        self.assertEqual(positions[-1], encoded.numel() - 2)

    def test_forced_alignment_covers_every_mel_frame(self) -> None:
        blank_id = 9
        targets = torch.tensor([5, 6], dtype=torch.long)
        logits = torch.full((7, 10), -8.0)
        sequence = [blank_id, 5, 5, blank_id, 6, 6, blank_id]
        for index, token in enumerate(sequence):
            logits[index, token] = 8.0
        log_probs = F.log_softmax(logits, dim=-1)
        result = forced_alignment_durations(
            log_probs,
            targets,
            blank_id=blank_id,
            mel_frames=13,
            frame_stride=2,
        )
        self.assertEqual(int(result.target_durations.sum().item()), 13)
        self.assertTrue(bool((result.target_durations > 0).all().item()))

    def test_repeated_target_uses_blank_transition(self) -> None:
        blank_id = 9
        targets = torch.tensor([5, 5], dtype=torch.long)
        logits = torch.full((5, 10), -8.0)
        sequence = [5, 5, blank_id, 5, 5]
        for index, token in enumerate(sequence):
            logits[index, token] = 8.0
        result = forced_alignment_durations(
            F.log_softmax(logits, dim=-1),
            targets,
            blank_id=blank_id,
            mel_frames=5,
            frame_stride=1,
        )
        self.assertEqual(int(result.target_durations.sum().item()), 5)
        self.assertEqual(result.target_durations.numel(), 2)

    def test_aligner_forward_and_ctc_loss_are_finite(self) -> None:
        frontend = SpanishTextFrontend()
        config = LykenoxCTCAlignerConfig(
            num_symbols=frontend.vocab_size,
            mel_bins=80,
            hidden_size=32,
            recurrent_layers=1,
        )
        model = LykenoxCTCAligner(config)
        mel = torch.randn(1, 80, 80)
        logits = model(mel)
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], frontend.vocab_size + 1)

        token_ids = torch.tensor(frontend.encode("hola"), dtype=torch.long)
        targets, _ = ctc_targets(token_ids)
        criterion = torch.nn.CTCLoss(blank=config.blank_id, zero_infinity=True)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = criterion(
            log_probs.transpose(0, 1),
            targets,
            torch.tensor([log_probs.shape[1]], dtype=torch.long),
            torch.tensor([targets.numel()], dtype=torch.long),
        )
        self.assertTrue(bool(torch.isfinite(loss).item()))
        loss.backward()

    def test_expand_content_durations_preserves_structural_zeros(self) -> None:
        frontend = SpanishTextFrontend()
        token_ids = torch.tensor(frontend.encode("si"), dtype=torch.long)
        _, positions = ctc_targets(token_ids)
        content = torch.tensor([4, 6], dtype=torch.long)
        full = expand_content_durations(token_ids, content, positions)
        self.assertEqual(int(full[0].item()), 0)
        self.assertEqual(int(full[-1].item()), 0)
        self.assertEqual(int(full.sum().item()), 10)


if __name__ == "__main__":
    unittest.main()
