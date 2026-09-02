"""Authoritative active source-path decision for the LYKENOX vocoder.

Historical codebook and continuous-source implementations remain evidence. This file states the
active engineering path after the 2026-09-02 held-out listening results.
"""

POLICY_ID = "LYX-POL-001"
DECISION_VERSION = "owned-vocoder-active-source-decision-v5-pitch-synchronous-cycle-source"

ACTIVE_SOURCE_ARCHITECTURE = "lykenox_owned_pitch_synchronous_residual_cycle_source_v1"
ACTIVE_SOURCE_MODEL = "lykenox_voice_engine/models/vocoder/network_pitch_synchronous_residual_cycle_source_v1.py"
ACTIVE_SOURCE_TRAINER = "lykenox_voice_engine/training/speech_vocoder_pitch_synchronous_cycle_source_train_v1.py"
ACTIVE_SOURCE_HELDOUT_RENDERER = "scripts/render_pitch_synchronous_cycle_source_v1.py"
ACTIVE_ONE_COMMAND_ENTRYPOINT = "scripts/train_pitch_synchronous_cycle_source_v1.py"

FIXED_MINIMUM_PHASE_RENDERER_RETAINED = True
FIXED_MINIMUM_PHASE_RENDERER_MODIFICATION_AUTHORIZED = False
STEP3F_REAL_RESIDUAL_IS_SOURCE_TRAINING_TARGET = True
SAMPLE_RATE_AUTOREGRESSIVE_MODEL = False

CONTINUOUS_SOURCE_V1_STATUS = "rejected_for_heldout_amplitude_collapse"
CONTINUOUS_SOURCE_V1_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.098, 0.128, 0.153)
CONTINUOUS_SOURCE_V1_CHECKPOINT_MAY_BE_USED_FOR_PRODUCT = False

# V2 remains the strongest positive learned source baseline: the owner reports good pronunciation,
# no gangoso and no final chillido on speech_0024, with the remaining dominant defect being robotic
# timbre. Its explicit residual-level factorization fixed the prior amplitude collapse.
CONTINUOUS_SOURCE_V2_STATUS = "best_positive_baseline_rejected_only_for_robotic_naturalness"
CONTINUOUS_SOURCE_V2_HELDOUT_PREDICTION_REFERENCE_RMS_RATIOS = (0.833, 0.905, 0.949)
CONTINUOUS_SOURCE_V2_SPEECH_0024_PRONUNCIATION_GOOD = True
CONTINUOUS_SOURCE_V2_SPEECH_0024_GANGOSO_PRESENT = False
CONTINUOUS_SOURCE_V2_SPEECH_0024_FINAL_CHILLIDO_PRESENT = False
CONTINUOUS_SOURCE_V2_SPEECH_0024_ROBOTIC_TIMBRE_PRESENT = True
CONTINUOUS_SOURCE_V2_IDENTITY_ROUNDTRIP_CEILING_CLEAN_AND_REFERENCE_LIKE = True
CONTINUOUS_SOURCE_V2_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False

# Coherent stochastic innovation was trained for 600 updates from the V2 warm start. Held-out
# listening on speech_0024 showed only a small reduction in robotization while gangoso returned and
# output level fell, making the net perceptual result worse than V2. The hypothesis that robotic
# timbre is primarily missing additive stochastic innovation is therefore rejected. No seed/mix/noise
# tuning is authorized.
COHERENT_INNOVATION_STATUS = "rejected_for_net_perceptual_regression"
COHERENT_INNOVATION_UPDATES = 600
COHERENT_INNOVATION_BEST_VAL_TOTAL = 2.5877118905385337
COHERENT_INNOVATION_ROBOTIZATION_CHANGE = "slightly_reduced"
COHERENT_INNOVATION_GANGOSO_CHANGE = "reintroduced"
COHERENT_INNOVATION_LEVEL_CHANGE = "decreased"
COHERENT_INNOVATION_NET_PROGRESS_POSITIVE = False
COHERENT_INNOVATION_CHECKPOINT_MAY_BE_USED_AS_FINAL_PRODUCT = False
FURTHER_COHERENT_INNOVATION_TUNING_AUTHORIZED = False

# Root interpretation after V2 and coherent-innovation listening:
# - exact Step3f residual + fixed renderer is clean;
# - V2 proves the learned source can preserve pronunciation and level without gangoso;
# - adding stochastic innovation does not solve naturalness;
# - the remaining learned target is a raw 512-sample vector on a fixed 256-sample frame grid even
#   though voiced residual fine structure is fundamentally periodic and phase-relative to F0.
# Regressing absolute sample phase against a frame grid creates a highly multimodal target and
# encourages phase-averaged/mechanical fine structure. The next representation removes that ambiguity.
ACTIVE_ROOT_CAUSE_HYPOTHESIS = "fixed_frame_grid_absolute_phase_ambiguity_in_voiced_real_residual_regression"
ACTIVE_ROOT_FIX = "learn_real_step3f_residual_cycles_in_pitch_synchronous_phase_coordinates"
PITCH_SYNCHRONOUS_REAL_CYCLE_EXTRACTION_AVAILABLE = True
PITCH_SYNCHRONOUS_REAL_CYCLE_SOURCE_IS_PARAMETRIC_ROSENBERG = False

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
METRICS_CAN_ACCEPT_PRODUCT_QUALITY = False
FULL_HELDOUT_LISTENING_REQUIRED = True
CPU_REFERENCE_DEVICE = True

NEXT_ACTION = "implement_and_train_pitch_synchronous_real_residual_cycle_source_then_listen_to_complete_heldout_audio"
