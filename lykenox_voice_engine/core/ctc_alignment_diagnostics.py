"""Diagnostics for explaining LYKENOX CTC duration ownership.

This module is training/audit-only. It decomposes a forced alignment into frames
that came directly from a CTC target state versus frames inherited from interior
CTC blank states. The final speech runtime does not need these diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lykenox_voice_engine.core.ctc_alignment import (
    CTCForcedAlignment,
    _assign_ctc_states,
)


@dataclass(frozen=True)
class CTCFrameOwnershipBreakdown:
    """Mel-frame ownership decomposition for one forced alignment."""

    frame_owners: torch.Tensor
    frame_is_direct_target: torch.Tensor
    direct_target_frames: torch.Tensor
    allocated_blank_frames: torch.Tensor

    @property
    def content_frames(self) -> torch.Tensor:
        return self.direct_target_frames + self.allocated_blank_frames


def ctc_frame_ownership_breakdown(
    alignment: CTCForcedAlignment,
    *,
    target_steps: int,
    frame_stride: int,
) -> CTCFrameOwnershipBreakdown:
    """Explain how each content duration was formed.

    ``direct_target_frames`` are frames whose original CTC state was the target
    phoneme itself. ``allocated_blank_frames`` are interior blank frames assigned
    to a neighboring target by the duration policy. Leading/trailing boundary
    blanks keep their negative boundary owner and therefore are not counted as
    content here.
    """

    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")

    state_path = alignment.state_path.detach().cpu().to(torch.long)
    step_owners = _assign_ctc_states(state_path, target_steps)
    step_is_direct_target = (state_path % 2) == 1

    frame_owners = step_owners.repeat_interleave(frame_stride)
    frame_is_direct_target = step_is_direct_target.repeat_interleave(frame_stride)

    mel_frames = int(alignment.mel_frames)
    if frame_owners.numel() < mel_frames:
        missing = mel_frames - frame_owners.numel()
        frame_owners = torch.cat((frame_owners, frame_owners[-1].repeat(missing)))
        frame_is_direct_target = torch.cat(
            (
                frame_is_direct_target,
                frame_is_direct_target[-1].repeat(missing),
            )
        )
    frame_owners = frame_owners[:mel_frames]
    frame_is_direct_target = frame_is_direct_target[:mel_frames]

    direct_mask = (frame_owners >= 0) & frame_is_direct_target
    blank_mask = (frame_owners >= 0) & (~frame_is_direct_target)

    direct_target_frames = torch.bincount(
        frame_owners[direct_mask],
        minlength=target_steps,
    ).to(torch.long)
    allocated_blank_frames = torch.bincount(
        frame_owners[blank_mask],
        minlength=target_steps,
    ).to(torch.long)

    reconstructed = direct_target_frames + allocated_blank_frames
    expected = alignment.target_durations.detach().cpu().to(torch.long)
    if not torch.equal(reconstructed, expected):
        raise RuntimeError(
            "CTC ownership breakdown does not reconstruct target durations"
        )

    return CTCFrameOwnershipBreakdown(
        frame_owners=frame_owners,
        frame_is_direct_target=frame_is_direct_target,
        direct_target_frames=direct_target_frames,
        allocated_blank_frames=allocated_blank_frames,
    )
