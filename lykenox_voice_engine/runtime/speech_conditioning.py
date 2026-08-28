"""Reference-free acoustic -> vocoder conditioning for LYKENOX Speech.

This module is product-side inference code, not a training target extractor.  The
persistent acoustic model predicts mel, F0 and voicing from text; this boundary converts
those predictions into the exact mel + F0 + voiced contract consumed by the accepted
LYKENOX v4.1 source-filter vocoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


SPEECH_VOCODER_CONDITIONING_VERSION = "speech-vocoder-conditioning-v1"

# The persistent speech acoustic model was supervised from lykenox-pitch-v1.  That
# extractor searches an integer-lag grid at 24 kHz with a nominal 60..350 Hz range, whose
# exact highest representable target is 24000 / floor(24000 / 350) = 352.941176... Hz.
# Keep product inference inside the support seen by the accepted acoustic/vocoder pair.
PREDICTED_SPEECH_F0_MIN_HZ = 60.0
PREDICTED_SPEECH_F0_MAX_HZ = 352.94117647058823
DEFAULT_VOICING_THRESHOLD = 0.5


@dataclass(frozen=True)
class SpeechVocoderConditioning:
    """Frame-aligned conditioning tensors ready for the LYKENOX speech vocoder."""

    mel: torch.Tensor
    f0_hz: torch.Tensor
    voiced: torch.Tensor
    voicing_probability: torch.Tensor
    raw_f0_hz: torch.Tensor
    f0_clipped_mask: torch.Tensor
    frame_mask: torch.Tensor


def prepare_speech_vocoder_conditioning(
    acoustic_output: dict[str, torch.Tensor],
    *,
    voicing_threshold: float = DEFAULT_VOICING_THRESHOLD,
) -> SpeechVocoderConditioning:
    """Convert text-only acoustic predictions into the accepted v4.1 input contract.

    Rules:
    - no waveform, reference speaker audio, or pitch target is accepted as input;
    - voicing is thresholded from the acoustic model's logits because the v4.1 vocoder was
      trained against binary voiced targets;
    - F0 is exactly zero on predicted-unvoiced/padded frames;
    - predicted-voiced F0 is clamped only to the pitch support used to train the persistent
      speech acoustic/vocoder pair;
    - mel/F0/voicing remain on the exact acoustic frame grid.
    """

    if not 0.0 < voicing_threshold < 1.0:
        raise ValueError("voicing_threshold must be strictly between zero and one")

    required = ("mel", "f0_prediction_hz", "voicing_logits", "mel_mask")
    missing = [key for key in required if key not in acoustic_output]
    if missing:
        raise KeyError(f"acoustic output is missing required conditioning tensors: {missing}")

    mel = acoustic_output["mel"]
    raw_f0 = acoustic_output["f0_prediction_hz"]
    logits = acoustic_output["voicing_logits"]
    frame_mask = acoustic_output["mel_mask"].bool()

    if mel.ndim != 3:
        raise ValueError("mel must have shape [batch, frames, mel_bins]")
    if raw_f0.shape != mel.shape[:2] or logits.shape != mel.shape[:2]:
        raise ValueError("F0/voicing predictions must match mel [batch, frames]")
    if frame_mask.shape != mel.shape[:2]:
        raise ValueError("mel_mask must match mel [batch, frames]")
    if not bool(torch.isfinite(mel).all()):
        raise ValueError("predicted mel contains non-finite values")
    if not bool(torch.isfinite(raw_f0).all()):
        raise ValueError("predicted F0 contains non-finite values")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("predicted voicing logits contain non-finite values")
    if not bool((raw_f0 >= 0.0).all()):
        raise ValueError("predicted F0 must be non-negative")

    probability = torch.sigmoid(logits)
    predicted_voiced = (probability >= voicing_threshold) & frame_mask
    clipped = torch.clamp(
        raw_f0,
        min=PREDICTED_SPEECH_F0_MIN_HZ,
        max=PREDICTED_SPEECH_F0_MAX_HZ,
    )
    f0_clipped_mask = predicted_voiced & (
        (raw_f0 < PREDICTED_SPEECH_F0_MIN_HZ)
        | (raw_f0 > PREDICTED_SPEECH_F0_MAX_HZ)
    )
    f0_hz = torch.where(predicted_voiced, clipped, torch.zeros_like(clipped))
    voiced = predicted_voiced.to(mel.dtype)

    frame_scale = frame_mask.to(mel.dtype)
    mel = mel * frame_scale.unsqueeze(-1)
    probability = probability * frame_scale
    raw_f0 = raw_f0 * frame_scale

    return SpeechVocoderConditioning(
        mel=mel,
        f0_hz=f0_hz,
        voiced=voiced,
        voicing_probability=probability,
        raw_f0_hz=raw_f0,
        f0_clipped_mask=f0_clipped_mask,
        frame_mask=frame_mask,
    )
