"""Authoritative active source-path decision for the LYKENOX vocoder.

Historical codebook and continuous-source implementations remain evidence.  This file states the
active engineering path after the 2026-09-02 held-out listening result for level-factored V2.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v3-coherent-stochastic-innovation"

ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_coherent_innovation_residual_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_minimum_phase_coherent_innovation_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_coherent_innovation_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_coherent_innovation_source_v1.py"
ACTIVE_ONE_COMMAND_ENTRYPOINT = "scripts/train_coherent_innovation_source_v1.py"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
RESIDUAL_512_256_SQRT_HANN_REPRESENTATION_RETAINED = True
FRAME_RATE_AUTOREGRESSIVE_COHERENT_SOURCE = True
EXPLICIT_RESIDUAL_LOG_RMS_HEAD_RETAINED = True
STOCHASTIC_APERIODIC_INNOVATION_BRANCH = True
INNOVATION_SPECTRAL_COLOR_IS_FRAME_RATE_PREDICTED = True
INNOVATION_IS_NOT_RECURRENT_FEEDBACK = True
SAMPLE_RATE_AUTOREGRESSIVE_MODEL = False

CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V1_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.098, 0.128, 0.153)
CONTINUOUS_SOURCE_V1_CHECKPOINT_MAY_BE_USED_FOR_PRODUCT = False

# V2 is a genuine positive result but not product-acceptable: it fixed the amplitude collapse and
# the owner reports that speech_0024 has good pronunciation with no gangoso and no chillido in the
# final rendered waveform.  The remaining dominant audible defect is robotic timbre.  The isolated
# predicted residual sounding like a chillido is explicitly not a voice-quality rejection criterion.
CONTINUOUS_SOURCE_V2_STATUS = "positive_intermediate_rejected_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
CONTINUOUS_SOURCE_V2_SPEECH_0024_PRONUNCIATION_GOOD = True
CONTINUOUS_SOURCE_V2_SPEECH_0024_GANGOSO_PRESENT = False
CONTINUOUS_SOURCE_V2_SPEECH_0024_FINAL_CHILLIDO_PRESENT = False
CONTINUOUS_SOURCE_V2_SPEECH_0024_ROBOTIC_TIMBRE_PRESENT = True
CONTINUOUS_SOURCE_V2_IDENTITY_ROUNDTRIP_CEILING_CLEAN_AND_REFERENCE_LIKE = True
CONTINUOUS_SOURCE_V2_IS_VALID_WARM_START_FOR_OWNED_SUCCESSOR = True
CONTINUOUS_SOURCE_V2_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

# Root interpretation: after level and envelope were exculpated, V2 still maps one acoustic
# conditioning trajectory to one deterministic residual fine-structure trajectory.  Aperiodic
# residual components are not uniquely predictable from mel/F0/voicing/periodicity, so forcing them
# through deterministic regression encourages mean/over-regularized fine structure and robotic tone.
ACTIVE_ROOT_CAUSE_HYPOTHESIS = "deterministic_regression_of_aperiodic_residual_fine_structure"
ACTIVE_ROOT_FIX = "separate_coherent_residual_trajectory_from_stochastic_aperiodic_innovation"
V2_OWNED_WARM_START_AUTHORIZED = True

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
VAL_ALLOWED_FOR_REJECTION_AND_CHECKPOINT_SELECTION = True
COMPLETE_HELDOUT_FREE_RUNNING_VALIDATION_REQUIRED = True
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

NEXT_ACTION = "train_coherent_innovation_source_from_v2_warm_start_then_listen_for_robotic_timbre_reduction"
