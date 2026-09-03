from __future__ import annotations

import torch

from lykenox_voice_engine.training.speech_pitch_conditioning_v2 import (
    PITCH_CONDITIONING_V2,
    PitchConditioningV2,
)
from lykenox_voice_engine.training.speech_vocoder_continuous_residual_source_v2_pitch_conditioning_v2 import (
    CHECKPOINT_SCHEMA_VERSION,
    RUN_DIR_NAME,
    _segment_with_conditioning_v2,
)
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_full_utterance_data import (
    OwnedVocoderUtterance,
)


def _utterance(frames: int = 4) -> OwnedVocoderUtterance:
    return OwnedVocoderUtterance(
        split="train",
        utterance_id="synthetic",
        wav_path="synthetic.wav",
        mel_frames=frames,
        mel=torch.zeros(frames, 80),
        f0_hz=torch.zeros(frames),
        voiced=torch.zeros(frames),
        periodicity=torch.full((frames,), 0.99),
        waveform=torch.zeros(frames * 256),
    )


def _conditioning() -> PitchConditioningV2:
    return PitchConditioningV2(
        f0_track_hz=torch.tensor([100.0, 110.0, 120.0, 130.0]),
        periodic_strength=torch.tensor([0.10, 0.20, 0.30, 0.40]),
        energy_confidence=torch.tensor([0.50, 0.60, 0.70, 0.80]),
        raw_periodicity=torch.tensor([0.70, 0.71, 0.72, 0.73]),
        anchor_voiced=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        frame_rms=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )


def _target(frames: int = 4) -> dict[str, torch.Tensor]:
    return {
        "residual_vectors": torch.zeros(frames + 1, 512),
        "residual": torch.zeros(frames * 256),
        "cepstrum": torch.zeros(frames, 64),
    }


def test_controlled_segment_replaces_only_three_conditioning_slots():
    tensors = _segment_with_conditioning_v2(
        _utterance(),
        _conditioning(),
        _target(),
        start=1,
        frames=2,
    )
    mel, f0_slot, second_slot, third_slot, target_vectors, residual, cepstrum, reference = tensors
    assert mel.shape == (1, 2, 80)
    assert torch.equal(f0_slot, torch.tensor([[110.0, 120.0]]))
    assert torch.equal(second_slot, torch.tensor([[0.60, 0.70]]))
    assert torch.equal(third_slot, torch.tensor([[0.20, 0.30]]))
    assert target_vectors.shape == (1, 3, 512)
    assert residual.shape == (1, 512)
    assert cepstrum.shape == (1, 2, 64)
    assert reference.shape == (1, 512)


def test_legacy_binary_voiced_and_raw_periodicity_do_not_leak_into_new_slots():
    utterance = _utterance()
    tensors = _segment_with_conditioning_v2(
        utterance,
        _conditioning(),
        _target(),
        start=0,
        frames=4,
    )
    assert not torch.equal(tensors[2], utterance.voiced.unsqueeze(0))
    assert not torch.equal(tensors[3], utterance.periodicity.unsqueeze(0))


def test_controlled_run_has_separate_checkpoint_identity():
    assert PITCH_CONDITIONING_V2 == "lykenox-pitch-conditioning-v2-continuous-strength"
    assert RUN_DIR_NAME == "continuous_residual_source_v2_pitch_conditioning_v2"
    assert "pitch-conditioning-v2" in CHECKPOINT_SCHEMA_VERSION
