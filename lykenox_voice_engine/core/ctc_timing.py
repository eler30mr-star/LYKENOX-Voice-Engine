"""LYKENOX timing policy for converting a CTC path into model-token durations.

The CTC aligner is intentionally not retrained when timing ownership policy changes.
This module interprets the already validated monotonic state path and keeps silence-like
blank runs away from spoken phonemes whenever the text stream provides a proper timing
carrier:

- leading blank -> <bos>
- trailing blank -> <eos>
- blank between words -> <wb>
- blank adjacent to an explicit punctuation pause -> that pause token
- intra-word blank -> split between the neighboring phonemes

The last case preserves coarticulation where there is no explicit silence token. It also
keeps genuinely suspicious intra-word alignments visible to the outlier gate instead of
inventing a hidden timing token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lykenox_voice_engine.core.ctc_alignment import CTCForcedAlignment
from lykenox_voice_engine.core.spanish_text_frontend import vocabulary


@dataclass(frozen=True)
class CTCTimingDurations:
    """Full model-token durations plus an auditable blank-ownership summary."""

    durations: torch.Tensor
    direct_target_frames: torch.Tensor
    leading_boundary_frames: int
    trailing_boundary_frames: int
    word_boundary_blank_frames: int
    pause_blank_frames: int
    neighbor_split_blank_frames: int

    @property
    def accounted_frames(self) -> int:
        return int(self.durations.sum().item())


def _frame_state_path(alignment: CTCForcedAlignment, frame_stride: int) -> torch.Tensor:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    states = alignment.state_path.detach().cpu().to(torch.long).repeat_interleave(frame_stride)
    mel_frames = int(alignment.mel_frames)
    if states.numel() < mel_frames:
        states = torch.cat((states, states[-1].repeat(mel_frames - states.numel())))
    return states[:mel_frames]


def expand_alignment_timing_durations(
    token_ids: torch.Tensor,
    positions: list[int],
    alignment: CTCForcedAlignment,
    *,
    frame_stride: int,
) -> CTCTimingDurations:
    """Map one legal CTC path to exact LYKENOX model-token durations.

    ``positions`` maps CTC target indices back into the complete frontend token stream.
    Unlike the older half-split policy, an interior blank run that crosses ``<wb>`` is
    assigned to that word-boundary token instead of inflating its neighboring phonemes.
    """

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    if len(positions) < 1:
        raise ValueError("positions must contain at least one acoustic target")

    token_values = [int(value) for value in token_ids.detach().cpu().tolist()]
    target_steps = len(positions)
    states = _frame_state_path(alignment, frame_stride)
    if states.numel() != int(alignment.mel_frames):
        raise RuntimeError("Expanded CTC state path does not match mel frame count")

    durations = torch.zeros_like(token_ids.detach().cpu(), dtype=torch.long)
    direct_mask = (states % 2) == 1
    direct_target_indices = states[direct_mask] // 2
    if direct_target_indices.numel() == 0:
        raise RuntimeError("CTC path contains no direct target frames")
    if int(direct_target_indices.max().item()) >= target_steps:
        raise RuntimeError("CTC target index exceeds frontend target mapping")

    direct_target_frames = torch.bincount(
        direct_target_indices,
        minlength=target_steps,
    ).to(torch.long)
    if not bool((direct_target_frames > 0).all().item()):
        raise RuntimeError("At least one CTC acoustic target has no direct frame occupancy")
    for target_index, position in enumerate(positions):
        durations[position] += int(direct_target_frames[target_index].item())

    vocab = vocabulary()
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    wb_id = vocab["<wb>"]
    pause_ids = {
        vocab[name]
        for name in ("<pau_short>", "<pau_long>")
        if name in vocab
    }
    try:
        bos_position = token_values.index(bos_id)
        eos_position = len(token_values) - 1 - token_values[::-1].index(eos_id)
    except ValueError as error:
        raise RuntimeError("Timing policy requires BOS and EOS tokens") from error

    leading = 0
    trailing = 0
    word_boundary = 0
    pause_blank = 0
    neighbor_split = 0

    cursor = 0
    total_frames = int(states.numel())
    while cursor < total_frames:
        if int(states[cursor].item()) % 2 == 1:
            cursor += 1
            continue

        start = cursor
        while cursor < total_frames and int(states[cursor].item()) % 2 == 0:
            cursor += 1
        end = cursor
        run_frames = end - start

        if start == 0:
            durations[bos_position] += run_frames
            leading += run_frames
            continue
        if end == total_frames:
            durations[eos_position] += run_frames
            trailing += run_frames
            continue

        left_state = int(states[start - 1].item())
        right_state = int(states[end].item())
        if left_state % 2 != 1 or right_state % 2 != 1:
            raise RuntimeError("Interior CTC blank run lacks direct target neighbors")
        left_target = left_state // 2
        right_target = right_state // 2
        if left_target >= target_steps or right_target >= target_steps:
            raise RuntimeError("Interior CTC blank neighbor exceeds target mapping")

        left_position = positions[left_target]
        right_position = positions[right_target]
        if right_position <= left_position:
            raise RuntimeError("Frontend target positions are not monotonic")

        wb_positions = [
            position
            for position in range(left_position + 1, right_position)
            if token_values[position] == wb_id
        ]
        left_token = token_values[left_position]
        right_token = token_values[right_position]

        if wb_positions:
            # A normal frontend stream has one <wb> here. If future versions expose
            # more than one, the first remains the deterministic timing carrier.
            durations[wb_positions[0]] += run_frames
            word_boundary += run_frames
            continue

        left_is_pause = left_token in pause_ids
        right_is_pause = right_token in pause_ids
        if left_is_pause or right_is_pause:
            if left_is_pause and right_is_pause:
                left_frames = (run_frames + 1) // 2
                durations[left_position] += left_frames
                durations[right_position] += run_frames - left_frames
            elif left_is_pause:
                durations[left_position] += run_frames
            else:
                durations[right_position] += run_frames
            pause_blank += run_frames
            continue

        # No textual silence carrier exists inside this word. Preserve the old
        # local coarticulation assumption, while keeping long cases visible to audit.
        left_frames = (run_frames + 1) // 2
        durations[left_position] += left_frames
        durations[right_position] += run_frames - left_frames
        neighbor_split += run_frames

    if int(durations.sum().item()) != int(alignment.mel_frames):
        raise RuntimeError("Timing durations do not cover every mel frame exactly")

    return CTCTimingDurations(
        durations=durations,
        direct_target_frames=direct_target_frames,
        leading_boundary_frames=leading,
        trailing_boundary_frames=trailing,
        word_boundary_blank_frames=word_boundary,
        pause_blank_frames=pause_blank,
        neighbor_split_blank_frames=neighbor_split,
    )
