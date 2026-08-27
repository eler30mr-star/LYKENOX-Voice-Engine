from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.core.ctc_alignment import CTCForcedAlignment, ctc_targets
from lykenox_voice_engine.core.ctc_timing import expand_alignment_timing_durations
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend


class LykenoxCTCTimingTests(unittest.TestCase):
    @staticmethod
    def _alignment(state_path: list[int]) -> CTCForcedAlignment:
        return CTCForcedAlignment(
            state_path=torch.tensor(state_path, dtype=torch.long),
            target_durations=torch.tensor([3, 3], dtype=torch.long),
            leading_boundary_frames=0,
            trailing_boundary_frames=0,
            score=-1.0,
            score_per_step=-0.1,
            downsampled_steps=len(state_path),
            mel_frames=len(state_path),
        )

    def test_word_boundary_blank_is_not_assigned_to_phonemes(self) -> None:
        frontend = SpanishTextFrontend()
        token_ids = torch.tensor(frontend.encode("a e"), dtype=torch.long)
        _, positions = ctc_targets(token_ids)
        self.assertEqual(len(positions), 2)

        timing = expand_alignment_timing_durations(
            token_ids,
            positions,
            self._alignment([1, 1, 2, 2, 3, 3]),
            frame_stride=1,
        )
        vocab = frontend.vocabulary()
        values = token_ids.tolist()
        wb_position = values.index(vocab["<wb>"])

        self.assertEqual(int(timing.durations[positions[0]].item()), 2)
        self.assertEqual(int(timing.durations[wb_position].item()), 2)
        self.assertEqual(int(timing.durations[positions[1]].item()), 2)
        self.assertEqual(timing.word_boundary_blank_frames, 2)
        self.assertEqual(timing.neighbor_split_blank_frames, 0)
        self.assertEqual(timing.accounted_frames, 6)

    def test_intra_word_blank_still_splits_between_neighbors(self) -> None:
        frontend = SpanishTextFrontend()
        token_ids = torch.tensor(frontend.encode("ae"), dtype=torch.long)
        _, positions = ctc_targets(token_ids)
        timing = expand_alignment_timing_durations(
            token_ids,
            positions,
            self._alignment([1, 1, 2, 2, 3, 3]),
            frame_stride=1,
        )

        self.assertEqual(int(timing.durations[positions[0]].item()), 3)
        self.assertEqual(int(timing.durations[positions[1]].item()), 3)
        self.assertEqual(timing.word_boundary_blank_frames, 0)
        self.assertEqual(timing.neighbor_split_blank_frames, 2)
        self.assertEqual(timing.accounted_frames, 6)

    def test_boundary_and_word_blank_frames_all_remain_accounted(self) -> None:
        frontend = SpanishTextFrontend()
        token_ids = torch.tensor(frontend.encode("a e"), dtype=torch.long)
        _, positions = ctc_targets(token_ids)
        timing = expand_alignment_timing_durations(
            token_ids,
            positions,
            self._alignment([0, 1, 2, 3, 4]),
            frame_stride=1,
        )
        vocab = frontend.vocabulary()
        values = token_ids.tolist()
        wb_position = values.index(vocab["<wb>"])

        self.assertEqual(timing.leading_boundary_frames, 1)
        self.assertEqual(timing.trailing_boundary_frames, 1)
        self.assertEqual(timing.word_boundary_blank_frames, 1)
        self.assertEqual(int(timing.durations[0].item()), 1)
        self.assertEqual(int(timing.durations[wb_position].item()), 1)
        self.assertEqual(int(timing.durations[-1].item()), 1)
        self.assertEqual(timing.accounted_frames, 5)


if __name__ == "__main__":
    unittest.main()
