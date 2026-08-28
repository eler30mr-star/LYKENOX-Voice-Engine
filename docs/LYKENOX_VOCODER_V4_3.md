# LYKENOX Vocoder v4.3 — carrier-filter corrective gate

## Why v4.3 exists

Persistent v4.2 training improved full-utterance oracle reconstruction substantially versus v4.1, but listening still found a residual periodic metallic/insect-like chillido. The subsequent trained source-path ablation established the critical causal fact:

- reducing `source_path_gain` mainly reduces useful voice, RMS, centroid and upper-band detail;
- normalized periodic character remains similar through the useful gain range;
- at `source_path_gain = 0.0`, useful speech collapses to a weak subgrave residual.

Therefore a runtime source-gain tweak is rejected. V4.2 remains too dependent on the transformed source branch as part of the waveform itself.

## Corrective principle

V4.3 changes the authority boundary:

```text
carrier = F0 / phase / aperiodic excitation
mel     = timbre / spectral envelope / amplitude filtering authority
```

The carrier is no longer added to a separate mel waveform path. Every audible sample must originate from the carrier and pass through a mel-controlled nonlinear filter.

Architecture identity:

```text
lykenox_mel_filtered_carrier_v4_3
```

Core structure:

```text
mel [frame rate]
  -> 96-channel conditioning encoder
  -> deterministic interpolation
  -> multiplicative filter controls only

F0 + voicing
  -> deterministic anti-aliased 24-harmonic carrier
  + deterministic aperiodic excitation
  + voiced/log-F0 cues
  -> bias-free 64-channel carrier stem
  -> 8 bias-free dilated carrier-filter blocks
     dilations: 1,2,4,8,16,32,64,128
     mel controls multiplicative gain only
  -> bias-free waveform projection
  -> fixed 30 Hz HPF
  -> tanh waveform
```

No transposed convolution, learned temporal upsampling, external vocoder, reference audio, source speaker, voice conversion, cloud dependency or runtime model download is introduced.

## Structural invariant: no additive mel shortcut

The waveform path is bias-free and mel enters only through multiplicative gains. Therefore:

```text
carrier = 0
mel != 0
=> waveform == 0 exactly
```

This prevents the v4.2 failure mode where mel and source paths can split waveform authority in a way that leaves periodic source leakage exposed.

## Richer carrier and local spectral-contrast objective

V4.1/v4.2 used eight explicit harmonics. Removing harmonics degraded useful voice, so v4.3 instead uses 24 deterministic harmonics with `1/sqrt(h)` weighting, total harmonic RMS normalization and a smooth anti-alias guard. The carrier is not a direct waveform; it must pass through the mel-conditioned filter.

V4.3 also adds the training-only target-relative local log-STFT contrast objective:

```text
vocoder-local-spectral-contrast-v1
```

It compares narrow local peak/notch structure against the paired real waveform instead of globally suppressing natural harmonic speech structure.

## Architecture smoke — PASSED

The bounded real-data CPU smoke passed:

```text
status: pass
architecture: lykenox_mel_filtered_carrier_v4_3
persistent_training_started: false
v4_2_checkpoint_mutated: false
exact_length_contract: true
structural_finite: true
no_additive_mel_shortcut: true
zero_carrier_max_abs: 0.0
gradients_finite: true
total_decreased: true
envelope_decreased: true
local_spectral_contrast_decreased: true
parameter_budget_pass: true
receptive_field_pass: true
```

Measured CPU feasibility:

```text
parameters: 334272
receptive_field_samples: 1559
receptive_field_ms: 64.958
mean_seconds_per_step: 0.7829
max_seconds_per_step: 1.0419
benchmark_audio_seconds: 1.024
benchmark_inference_seconds_median: 0.4167
benchmark_rtf: 0.4069
```

Bounded optimization reduced all required corrective objectives:

```text
probe_before.total: 6.311642
probe_after.total: 2.628071
probe_before.log_mel_envelope: 3.790301
probe_after.log_mel_envelope: 1.280633
probe_before.local_spectral_contrast: 0.323659
probe_after.local_spectral_contrast: 0.302628
```

## Persistent trainer contract

The separate v4.3 checkpoint/trainer implementation is:

```text
lykenox_voice_engine/training/speech_vocoder_v4_3_artifact.py
lykenox_voice_engine/training/speech_vocoder_v4_3_train.py
```

Trainer identity:

```text
v4-3-bounded-resumable-v1
```

Production artifact directory:

```text
models/lykenox_identity/training/vocoder_mel_filtered_carrier_v4_3/
```

Stable target-referenced validation selection score:

```text
reconstruction
+ 0.50 * envelope
+ 0.25 * spectral_balance
+ 0.20 * local_spectral_contrast
```

Adversarial and feature-matching terms may train the generator after warmup, but they are excluded from checkpoint selection.

Default persistent configuration:

```text
segment_mel_frames: 64
train_items: 118
val_items: 14
max_epochs: 28
warmup_epochs: 4
patience: 6
seed: 2430
generator_lr: 2e-4
discriminator_lr: 1e-4
envelope_weight: 0.50
balance_weight: 0.25
contrast_weight: 0.20
adversarial_weight: 0.03
feature_matching_weight: 0.50
gradient_clip_norm: 5.0
checkpoint_every_updates: 8
time_budget_seconds: 70
```

## Exact-resume gate — PASSED

The exact-resume smoke compared four deterministic updates executed directly against `2 + checkpoint/reload + 2` and passed every persistent-state equality check:

```text
status: pass
trainer_contract_version: v4-3-bounded-resumable-v1
global_step_exact: true
epoch_exact: true
next_item_offset_exact: true
generator_state_exact: true
discriminator_state_exact: true
generator_optimizer_exact: true
discriminator_optimizer_exact: true
torch_rng_state_exact: true
run_config_exact: true
v4_2_checkpoint_unchanged: true
temporary_artifacts_removed: true
persistent_v4_3_training_started: false
next_gate: start_bounded_resumable_v4_3_persistent_training
```

This closes the architecture and resumability gates. **Persistent v4.3 training is now authorized.**

## Authorized persistent run

Run only the default training identity:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_3_train
```

If an invocation returns:

```text
status: incomplete
next_gate: rerun_same_command_to_resume
```

rerun the identical command. Do not change hyperparameters, delete `last.pt`, replace checkpoints, or restart the experiment under the same artifact directory.

Numerical completion requires:

```text
persistent_training_complete: true
training_improved: true
envelope_improved: true
local_spectral_contrast_improved: true
status: pass
```

Completion by `early_stopping` or `max_epochs` is acceptable if those gates pass.

## Mandatory post-training gate

A numerical PASS does not grant perceptual or product acceptance. After training, the next mandatory gate is full held-out oracle generation using target mel + target F0 + target voicing. The characteristic metallic/chillido artifact must be materially resolved on complete utterances before predicted acoustic conditioning is reconnected.

Remaining order:

```text
v4.3 architecture smoke          [PASS]
  -> exact-resume trainer gate   [PASS]
  -> bounded persistent v4.3 training [AUTHORIZED]
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> full reference-free text-to-waveform perceptual gate
```

No `/speak` integration or product-runtime acceptance is authorized before the full-utterance oracle waveform gate passes.
