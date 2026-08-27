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


@dataclass(frozen=True)
class CTCForcedAlignment:
    """One monotonic hard alignment from CTC frame posteriors."""

    state_path: torch.Tensor
    target_durations: torch.Tensor
    score: float
    score_per_step: float
    downsampled_steps: int
    mel_frames: int


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
    """Find the best legal CTC state sequence with a monotonic Viterbi search.

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

    target_values = [int(value) for value in targets_cpu.tolist()]
    expanded: list[int] = [blank_id]
    for token in target_values:
        expanded.extend((token, blank_id))
    state_count = len(expanded)

    previous = [float("-inf")] * state_count
    previous[0] = float(probabilities[0, blank_id].item())
    previous[1] = float(probabilities[0, target_values[0]].item())
    backpointer: list[list[int]] = [[0] * state_count for _ in range(time_steps)]

    for time_index in range(1, time_steps):
        current = [float("-inf")] * state_count
        for state in range(state_count):
            candidates: list[tuple[float, int]] = [(previous[state], 0)]
            if state > 0:
                candidates.append((previous[state - 1], 1))
            if (
                state > 1
                and expanded[state] != blank_id
                and expanded[state] != expanded[state - 2]
            ):
                candidates.append((previous[state - 2], 2))
            best_score, move = max(candidates, key=lambda item: item[0])
            current[state] = best_score + float(probabilities[time_index, expanded[state]].item())
            backpointer[time_index][state] = move
        previous = current

    final_states = (state_count - 1, state_count - 2)
    state = max(final_states, key=lambda index: previous[index])
    final_score = previous[state]
    if not math.isfinite(final_score):
        raise RuntimeError("No finite CTC alignment path found")

    path = torch.empty((time_steps,), dtype=torch.long)
    path[-1] = state
    for time_index in range(time_steps - 1, 0, -1):
        state -= backpointer[time_index][state]
        path[time_index - 1] = state

    return path, final_score


def _assign_blank_states(state_path: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Assign every CTC timestep to a neighboring target token.

    Target states are odd indices in the expanded CTC graph. Blank runs are split
    between neighboring acoustic tokens so derived durations cover every acoustic
    frame instead of dropping inter-symbol silence.
    """

    assignments = torch.full_like(state_path, -1)
    target_mask = (state_path % 2) == 1
    assignments[target_mask] = state_path[target_mask] // 2

    total = int(assignments.numel())
    cursor = 0
    while cursor < total:
        if int(assignments[cursor].item()) >= 0:
            cursor += 1
            continue
        start = cursor
        while cursor < total and int(assignments[cursor].item()) < 0:
            cursor += 1
        end = cursor

        left = int(assignments[start - 1].item()) if start > 0 else -1
        right = int(assignments[end].item()) if end < total else -1

        if left < 0 and right < 0:
            raise RuntimeError("Blank-only path cannot be converted to token durations")
        if left < 0:
            assignments[start:end] = right
        elif right < 0:
            assignments[start:end] = left
        else:
            run = end - start
            left_count = (run + 1) // 2
            assignments[start : start + left_count] = left
            assignments[start + left_count : end] = right

    if int(assignments.min().item()) < 0 or int(assignments.max().item()) >= target_steps:
        raise RuntimeError("Invalid token assignment produced from CTC state path")
    return assignments


def forced_alignment_durations(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    mel_frames: int,
    frame_stride: int,
) -> CTCForcedAlignment:
    """Convert a CTC Viterbi path into exact mel-frame durations per target."""

    if mel_frames < 1:
        raise ValueError("mel_frames must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")

    path, score = ctc_viterbi_state_path(log_probs, targets, blank_id)
    downsampled_assignment = _assign_blank_states(path, int(targets.numel()))
    frame_assignment = downsampled_assignment.repeat_interleave(frame_stride)
    if frame_assignment.numel() < mel_frames:
        pad = frame_assignment[-1].repeat(mel_frames - frame_assignment.numel())
        frame_assignment = torch.cat((frame_assignment, pad))
    frame_assignment = frame_assignment[:mel_frames]
    durations = torch.bincount(frame_assignment, minlength=int(targets.numel())).to(torch.long)

    if int(durations.sum().item()) != mel_frames:
        raise RuntimeError("Forced-alignment durations do not cover all mel frames")

    steps = int(path.numel())
    return CTCForcedAlignment(
        state_path=path,
        target_durations=durations,
        score=score,
        score_per_step=score / max(1, steps),
        downsampled_steps=steps,
        mel_frames=mel_frames,
    )


def expand_content_durations(
    token_ids: torch.Tensor,
    content_durations: torch.Tensor,
    positions: list[int],
) -> torch.Tensor:
    """Map CTC acoustic durations back to the full model token sequence."""

    if content_durations.ndim != 1:
        raise ValueError("content_durations must be one-dimensional")
    if len(positions) != int(content_durations.numel()):
        raise ValueError("positions and content_durations must have matching lengths")

    durations = torch.zeros_like(token_ids, dtype=torch.long)
    for position, duration in zip(positions, content_durations.tolist(), strict=True):
        durations[position] = int(duration)
    return durations
