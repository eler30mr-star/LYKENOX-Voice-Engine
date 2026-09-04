"""Authoritative active dataset gate for LYKENOX speech/vocoder work.

This file supersedes only the dataset-quality / next-data-action section of
speech_vocoder_active_source_decision.py v20. All source-architecture, renderer, phase-isolation,
and historical forensic conclusions in that file remain unchanged.

Policy: LYX-POL-001 v1.1.
"""

POLICY_ID = "LYX-POL-001"
DATASET_DECISION_VERSION = "owned-vocoder-active-dataset-decision-v1-recording-v2"
SUPERSEDES_SOURCE_DECISION_DATASET_GATE_VERSION = (
    "owned-vocoder-active-source-decision-v20-phase-isolated-clean-v1-gate"
)
SUPERSEDES_ONLY_DATASET_QUALITY_AND_NEXT_DATA_ACTION = True
SOURCE_ARCHITECTURE_DECISION_UNCHANGED = True
FIXED_MINIMUM_PHASE_RENDERER_UNCHANGED = True

OLD_132_CORPUS_STATUS = "preserved_forensic_not_primary_new_training_source"
AFFTDN_CALIBRATION_STATUS = "rejected_insufficient_cleanup_and_muffled_environment_effect"
DEEPFILTERNET_CALIBRATION_STATUS = (
    "good_voice_preservation_and_noise_isolation_but_some_external_events_escape"
)
CONTINUE_OPTIMIZING_OLD_132_CLEANUP_AUTHORIZED = False
OLD_132_DENOISE_BATCH_AUTHORIZED = False

RECORDING_V2_REQUIRED = True
RECORDING_V2_STATUS = "pending_clean_recapture"
RECORDING_V2_REUSE_EXISTING_132_TEXTS_AND_SPLITS = True
RECORDING_V2_REUSE_OLD_AUDIO = False
RECORDING_V2_RAW_MUST_REMAIN_IMMUTABLE = True
RECORDING_V2_EXTERNAL_EVENT_OVERLAP_REQUIRES_RETAKE = True
RECORDING_V2_HUMAN_AUDITORY_ACCEPTANCE_REQUIRED = True
RECORDING_V2_ALL_DERIVED_TARGETS_MUST_BE_REGENERATED = True
RECORDING_V2_GOLD_ORACLES_MUST_BE_RERUN = True

FURTHER_SOURCE_TRAINING_AUTHORIZED = False
FURTHER_SOURCE_ARCHITECTURE_CHANGES_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False
POSTHOC_VOCODER_DENOISE_AUTHORIZED = False
POSTHOC_VOCODER_EQ_AUTHORIZED = False

ACTIVE_DATASET_GATE = (
    "capture_audibly_validate_recording_v2_then_regenerate_all_acoustic_targets_and_rerun_gold_oracles"
)
NEXT_ACTION = "prepare_and_capture_recording_v2_session"
DECISION_DOCUMENT = "docs/LYKENOX_IDENTITY_VOICE_RECORDING_V2_DECISION_2026-09-04.md"
