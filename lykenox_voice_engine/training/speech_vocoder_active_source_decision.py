"""Authoritative active source-path decision for the LYKENOX vocoder.

Historical vocoder/codebook decision modules remain immutable evidence. This file states only the
current engineering direction after the 2026-09-02 root-cause work.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v1-continuous-residual"

ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_continuous_residual_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_minimum_phase_continuous_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_continuous_residual_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_continuous_residual_source_v1.py"
ACTIVE_ARCHITECTURE_DOC = "docs/LYKENOX_VOCODER_CONTINUOUS_RESIDUAL_SOURCE_ARCHITECTURE.md"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
RESIDUAL_512_256_SQRT_HANN_REPRESENTATION_RETAINED = True
FRAME_RATE_AUTOREGRESSIVE_CONTINUOUS_SOURCE = True
SAMPLE_RATE_AUTOREGRESSIVE_MODEL = False

DISCRETE_RESIDUAL_CODEBOOK_PRODUCT_PATH_CLOSED = True
CELP_CODEBOOK_SELECTOR_TRAINING_AUTHORIZED = False
FURTHER_CODEBOOK_RETENTION_SWEEPS_AUTHORIZED = False
FURTHER_CODEBOOK_PRESELECTION_TUNING_AUTHORIZED = False
FURTHER_CODEBOOK_BEAM_TUNING_AUTHORIZED = False
PARAMETRIC_ROSENBERG_SOURCE_REOPEN_AUTHORIZED = False

THIRD_PARTY_MODEL_OR_CHECKPOINT_AUTHORIZED = False
REMOTE_MODEL_OR_TTS_SERVICE_AUTHORIZED = False
POSTHOC_GAIN_NORMALIZATION_AUTHORIZED = False
POSTHOC_EQ_AUTHORIZED = False
POSTHOC_DENOISING_AUTHORIZED = False
PREDICTED_DURATION_MODIFICATION_AUTHORIZED = False

TRAIN_SPLIT_ONLY_FOR_OPTIMIZER_UPDATES = True
VAL_ALLOWED_FOR_REJECTION_ONLY = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

# This is no longer an architecture-search gate. The next execution is the actual owned source
# training run followed by complete held-out listening of the resulting source path.
NEXT_ACTION = "train_owned_continuous_residual_source_then_render_complete_heldout_audio"
