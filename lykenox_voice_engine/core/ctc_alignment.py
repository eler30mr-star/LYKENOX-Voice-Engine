"""LYKENOX-owned CTC forced-alignment utilities for speech training.

The final product does not need this aligner at inference time. It is a training
component used to derive real frame durations from the owner's audio and text
without relying on an external TTS/ASR executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from lykenox_voice_engine.core.spanish_text_frontend import vocabulary


_LEADING_BOUNDARY = -2
_TRAILING_BOUNDARY = -3
_UNASSIGNED = -1


@dataclass(frozen=True)
class CTCForcedAlignment:
    """One monotonic hard alignment from CTC frame posteriors.

    ``target_durations`` contains only acoustic content/pause targets. Leading and
    trailing CTC blank runs are reported separately so recording-boundary silence
    is not folded into the first/last phoneme. Those boundary frames can be mapped
    to BOS/EOS by ``expand_content_durations``.
    """

    state_path: torch.Tensor
    target_durations: torch.Tensor
    leading_boundary_frames: int
    trailing_boundary_frames: int
    score: float
    score_per_step: float
    downsampled_steps: int
    mel_frames: int

    @property
    def accounted_frames(self) -> int:
        return (
            int(self.target_durations.sum().item())
            + self.leading_boundary_frames
            + self.trailing_boundary_frames
        )


def ctc_target_positions(token_ids: torch.Tensor) -> list[int]:
    """Return acoustic token positions that participate in CTC alignment.

    BOS/EOS/PAD are structural. ``<wb>`` marks lexical context for the text
    encoder but has no independent acoustic realization, so it is excluded from
    the forced-alignment target. Pause tokens remain alignable because they
    represent actual acoustic silence/prosodic time.
    """

    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    vocab = vocabulary()
    excluded_names = ("<pad>", "<bos>", "<eos>", "<wb>")
    excluded = {vocab[name] for name in excluded_names if name in vocab}
    values = token_ids.detach().cpu().tolist()
    return [index for index, value in enumerate(values) if int(value) not in excluded]


def ctc_targets(token_ids: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Extract the contiguous CTC target sequence and its original positions."""

    positions = ctc_target_positions(token_ids)
    if not positions:
        raise ValueError("No alignable content tokens")
    values = [int(token_ids[index].item()) for index in positions]
    return torch.tensor(values, dtype=torch.long, device=token_ids.device), positions


def minimum_ctc_steps(targets: torch.Tensor) -> int:
    """Minimum number of CTC timesteps required for a target sequence."""

    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("targets must be a non-empty one-dimensional tensor")
    repeats = int((targets[1:] == targets[:-1]).sum().item()) if targets.numel() > 1 else 0
    return int(targets.numel()) + repeats


def ctc_viterbi_state_path(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
) -> tuple[torch.Tensor, float]:
    """Find the best legal CTC state sequence with monotonic Viterbi search.

    The dynamic program is sequential in time but vectorized across all CTC
    states. The previous implementation used nested Python loops over time and
    state and performed ``.item()`` for every transition; that became the main
    bottleneck when regenerating alignments for the complete corpus on CPU.

    Args:
        log_probs: [time, classes] log probabilities.
        targets: [target_steps] token IDs, excluding structural tokens.
        blank_id: class ID reserved for the CTC blank.
    """

    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [time, classes]")
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("targets must be a non-empty one-dimensional tensor")
    if blank_id < 0 or blank_id >= log_probs.shape[1]:
        raise ValueError("blank_id is outside the log-probability class range")

    targets_cpu = targets.detach().cpu().to(torch.long)
    if int(targets_cpu.max().item()) >= blank_id:
        raise ValueError("target token IDs must be below the reserved blank ID")

    probabilities = log_probs.detach().cpu()
    time_steps = int(probabilities.shape[0])
    required = minimum_ctc_steps(targets_cpu)
    if time_steps < required:
        raise ValueError(f"CTC path impossible: {time_steps} frames for minimum {required}")

    target_count = int(targets_cpu.numel())
    state_count = target_count * 2 + 1
    expanded = torch.full((state_count,), blank_id, dtype=torch.long)
    expanded[1::2] = targets_cpu
    emissions = probabilities.index_select(1, expanded)

    previous = torch.full((state_count,), float("-inf"), dtype=emissions.dtype)
    previous[0] = emissions[0, 0]
    previous[1] = emissions[0, 1]

    # ``skip_allowed[s]`` means the state can be entered from s-2. CTC forbids
    # skipping into blank states and forbids the skip when repeated labels would
    # otherwise collapse without an intervening blank.
    skip_allowed = torch.zeros((state_count,), dtype=torch.bool)
    if state_count > 2:
        state_ids = torch.arange(state_count)
        skip_allowed = (
            (state_ids > 1)
            & (expanded != blank_id)
            & (expanded != torch.roll(expanded, shifts=2))
        )
        skip_allowed[:2] = False

    backpointer = torch.zeros((time_steps, state_count), dtype=torch.int8)
    neg_inf_vector = torch.full_like(previous, float("-inf"))

    for time_index in range(1, time_steps):
        stay = previous

        advance_one = torch.empty_like(previous)
        advance_one[0] = float("-inf")
        advance_one[1:] = previous[:-1]

        advance_two = neg_inf_vector.clone()
        if state_count > 2:
            advance_two[2:] = previous[:-2]
            advance_two = torch.where(skip_allowed, advance_two, neg_inf_vector)

        candidates = torch.stack((stay, advance_one, advance_two), dim=0)
        best_scores, moves = torch.max(candidates, dim=0)
        previous = best_scores + emissions[time_index]
        backpointer[time_index] = moves.to(torch.int8)

    final_candidates = torch.stack((previous[-1], previous[-2]))
    final_choice = int(torch.argmax(final_candidates).item())
    state = state_count - 1 if final_choice == 0 else state_count - 2
    final_score = float(previous[state].item())
    if not math.isfinite(final_score):
        raise RuntimeError("No finite CTC alignment path found")

    path = torch.empty((time_steps,), dtype=torch.long)
    path[-1] = state
    for time_index in range(time_steps - 1, 0, -1):
        state -= int(backpointer[time_index, state].item())
        path[time_index - 1] = state

    return path, final_score


def _assign_ctc_states(state_path: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Assign CTC steps without contaminating boundary phoneme durations.

    Odd CTC states are acoustic targets. Interior blank runs are divided between
    neighboring targets, preserving the previous monotonic policy. Blank runs
    before the first target and after the last target remain distinct boundary
    labels so they can supervise BOS/EOS silence instead of a spoken phoneme.
    """

    if state_path.ndim != 1 or state_path.numel() == 0:
        raise ValueError("state_path must be a non-empty one-dimensional tensor")
    if target_steps < 1:
        raise ValueError("target_steps must be positive")

    assignments = torch.full_like(state_path, _UNASSIGNED)
    target_mask = (state_path % 2) == 1
    target_positions = torch.nonzero(target_mask, as_tuple=False).flatten()
    if target_positions.numel() == 0:
        raise RuntimeError("CTC path contains no target states")

    assignments[target_mask] = state_path[target_mask] // 2
    first_target = int(target_positions[0].item())
    last_target = int(target_positions[-1].item())
    if first_target > 0:
        assignments[:first_target] = _LEADING_BOUNDARY
    if last_target + 1 < assignments.numel():
        assignments[last_target + 1 :] = _TRAILING_BOUNDARY

    cursor = first_target + 1
    while cursor < last_target:
        if int(assignments[cursor].item()) >= 0:
            cursor += 1
            continue
        start = cursor
        while cursor < last_target and int(assignments[cursor].item()) == _UNASSIGNED:
            cursor += 1
        end = cursor

        left = int(assignments[start - 1].item())
        right = int(assignments[end].item())
        if left < 0 or right < 0:
            raise RuntimeError("Interior CTC blank run is missing neighboring targets")
        run = end - start
        left_count = (run + 1) // 2
        assignments[start : start + left_count] = left
        assignments[start + left_count : end] = right

    if bool((assignments == _UNASSIGNED).any().item()):
        raise RuntimeError("Unassigned CTC states remain after boundary-aware mapping")
    content = assignments[assignments >= 0]
    if content.numel() == 0:
        raise RuntimeError("No acoustic target assignments produced")
    if int(content.max().item()) >= target_steps:
        raise RuntimeError("Invalid target index produced from CTC state path")
    return assignments


def forced_alignment_durations(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    mel_frames: int,
    frame_stride: int,
) -> CTCForcedAlignment:
    """Convert a CTC Viterbi path into exact boundary-aware mel-frame durations."""

    if mel_frames < 1:
        raise ValueError("mel_frames must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")

    path, score = ctc_viterbi_state_path(log_probs, targets, blank_id)
    downsampled_assignment = _assign_ctc_states(path, int(targets.numel()))
    frame_assignment = downsampled_assignment.repeat_interleave(frame_stride)
    if frame_assignment.numel() < mel_frames:
        pad = frame_assignment[-1].repeat(mel_frames - frame_assignment.numel())
        frame_assignment = torch.cat((frame_assignment, pad))
    frame_assignment = frame_assignment[:mel_frames]

    content_assignment = frame_assignment[frame_assignment >= 0]
    durations = torch.bincount(
        content_assignment,
        minlength=int(targets.numel()),
    ).to(torch.long)
    leading_boundary_frames = int((frame_assignment == _LEADING_BOUNDARY).sum().item())
    trailing_boundary_frames = int((frame_assignment == _TRAILING_BOUNDARY).sum().item())
    accounted = (
        int(durations.sum().item())
        + leading_boundary_frames
        + trailing_boundary_frames
    )
    if accounted != mel_frames:
        raise RuntimeError("Forced-alignment assignments do not cover all mel frames")

    steps = int(path.numel())
    return CTCForcedAlignment(
        state_path=path,
        target_durations=durations,
        leading_boundary_frames=leading_boundary_frames,
        trailing_boundary_frames=trailing_boundary_frames,
        score=score,
        score_per_step=score / max(1, steps),
        downsampled_steps=steps,
        mel_frames=mel_frames,
    )


def expand_content_durations(
    token_ids: torch.Tensor,
    content_durations: torch.Tensor,
    positions: list[int],
    *,
    leading_boundary_frames: int = 0,
    trailing_boundary_frames: int = 0,
) -> torch.Tensor:
    """Map acoustic and boundary durations back to the full model token sequence.

    Content/pause targets are restored at ``positions``. Leading recording silence
    is assigned to BOS and trailing recording silence to EOS. This preserves exact
    mel coverage without teaching the first or last spoken phoneme to emit silence.
    ``<wb>`` and other non-acoustic structural tokens keep zero duration.
    """

    if content_durations.ndim != 1:
        raise ValueError("content_durations must be one-dimensional")
    if len(positions) != int(content_durations.numel()):
        raise ValueError("positions and content_durations must have matching lengths")
    if leading_boundary_frames < 0 or trailing_boundary_frames < 0:
        raise ValueError("boundary frame counts must be non-negative")

    durations = torch.zeros_like(token_ids, dtype=torch.long)
    for position, duration in zip(positions, content_durations.tolist(), strict=True):
        durations[position] = int(duration)

    vocab = vocabulary()
    token_values = [int(value) for value in token_ids.detach().cpu().tolist()]
    if leading_boundary_frames:
        try:
            bos_position = token_values.index(vocab["<bos>"])
        except (KeyError, ValueError) as error:
            raise RuntimeError("Cannot assign leading boundary frames without BOS") from error
        durations[bos_position] = int(leading_boundary_frames)
    if trailing_boundary_frames:
        try:
            eos_position = len(token_values) - 1 - token_values[::-1].index(vocab["<eos>"])
        except (KeyError, ValueError) as error:
            raise RuntimeError("Cannot assign trailing boundary frames without EOS") from error
        durations[eos_position] = int(trailing_boundary_frames)
    return durations
