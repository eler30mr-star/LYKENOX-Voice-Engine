from __future__ import annotations

from pathlib import Path


def test_recording_v2_pilot_selector_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "prepare_identity_voice_recording_v2_pilot.py"
    text = path.read_text(encoding="utf-8")

    assert 'DEFAULT_PILOT_ITEMS = 10' in text
    assert 'TARGET_TRAIN_ITEMS = 6' in text
    assert 'TARGET_VAL_ITEMS = 4' in text
    assert 'speech_0021_6cd35984e877_seg_001' in text
    assert 'speech_0022_ba721f6129b9_seg_005' in text
    assert 'selection_metric_is_acceptance_evidence' in text
    assert 'audio_processed' in text
    assert 'training_authorized' in text
    assert 'pilot_manifest.csv' in text


def test_recording_v2_pilot_validation_does_not_process_audio() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "validate_identity_voice_recording_v2_pilot.py"
    text = path.read_text(encoding="utf-8")

    assert 'TARGET_SAMPLE_RATE = 48000' in text
    assert 'TARGET_CHANNELS = 1' in text
    assert 'VALID_SUBTYPES = {"PCM_24", "FLOAT"}' in text
    assert 'metrics_can_accept_perceptual_quality' in text
    assert 'human_auditory_acceptance_required' in text
    assert 'audio_processed' in text
    assert 'gain_normalization_used' in text
    assert 'denoise_used' in text
    assert 'pilot_technical_validation.json' in text
    assert 'pilot_listening_review.csv' in text

    # No write path may overwrite RAW captures.
    assert 'sf.write(' not in text
    assert 'torchaudio.save(' not in text
