from lykenox_voice_engine.training import speech_vocoder_active_dataset_decision as decision


def test_recording_v2_is_authoritative_dataset_gate() -> None:
    assert decision.POLICY_ID == "LYX-POL-001"
    assert decision.SUPERSEDES_ONLY_DATASET_QUALITY_AND_NEXT_DATA_ACTION is True
    assert decision.SOURCE_ARCHITECTURE_DECISION_UNCHANGED is True
    assert decision.FIXED_MINIMUM_PHASE_RENDERER_UNCHANGED is True
    assert decision.CONTINUE_OPTIMIZING_OLD_132_CLEANUP_AUTHORIZED is False
    assert decision.OLD_132_DENOISE_BATCH_AUTHORIZED is False
    assert decision.RECORDING_V2_REQUIRED is True
    assert decision.RECORDING_V2_REUSE_EXISTING_132_TEXTS_AND_SPLITS is True
    assert decision.RECORDING_V2_REUSE_OLD_AUDIO is False
    assert decision.RECORDING_V2_RAW_MUST_REMAIN_IMMUTABLE is True
    assert decision.RECORDING_V2_EXTERNAL_EVENT_OVERLAP_REQUIRES_RETAKE is True
    assert decision.RECORDING_V2_HUMAN_AUDITORY_ACCEPTANCE_REQUIRED is True
    assert decision.RECORDING_V2_ALL_DERIVED_TARGETS_MUST_BE_REGENERATED is True
    assert decision.RECORDING_V2_GOLD_ORACLES_MUST_BE_RERUN is True
    assert decision.FURTHER_SOURCE_TRAINING_AUTHORIZED is False
    assert decision.FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED is False
    assert decision.PREDICTED_DURATION_MODIFICATION_AUTHORIZED is False
    assert decision.POSTHOC_VOCODER_DENOISE_AUTHORIZED is False
    assert decision.POSTHOC_VOCODER_EQ_AUTHORIZED is False
    assert decision.NEXT_ACTION == "prepare_and_capture_recording_v2_session"
