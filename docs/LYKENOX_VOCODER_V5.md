# LYKENOX Vocoder v5 — non-sinusoidal corrective candidate

## Why v5 exists

The v4.x family isolated a persistent perceptual defect described as a radio-mistuned / metallic interference attached to the voice. Full-utterance oracle listening rejected v4.3 and v4.4 even when several target-referenced spectral metrics improved.

The decisive v4.4 path-attribution ablation showed that the explicit periodic branch preserved the defect much more strongly than the aperiodic branch. Reducing the harmonic bank from 24 to 8 helped somewhat but did not solve the problem. Therefore v5 is an architectural break rather than another v4.x tuning pass.

No further v4.x training is authorized.

## Architecture identity

```text
lykenox_stochastic_glottal_filter_v5
```

Source family:

```text
stochastic_glottal_pulse_noise
```

Explicit sinusoidal carrier:

```text
false
```

Deterministic harmonic bank:

```text
0 harmonics
```

F0 controls sample-rate glottal timing and phase features but does not create an audible sinusoidal bank. Broadband deterministic noise is shaped into stochastic voiced pulse bursts, a low voiced broadband floor and an unvoiced broadband component. Mel plus phase/F0 controls select and gate only bias-free excitation-dependent filters.

Structural invariant:

```text
excitation_scale = 0
mel != 0
F0 != 0
voiced != 0
=> waveform == 0 exactly
```

## Architecture smoke — PASSED

Local CPU smoke result:

```text
status: pass
architecture: lykenox_stochastic_glottal_filter_v5
source_family: stochastic_glottal_pulse_noise
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
persistent_training_started: false
historical_checkpoints_mutated: false
exact_length_contract: true
structural_finite: true
no_conditioning_only_waveform: true
zero_excitation_max_abs: 0.0
no_sinusoidal_carrier: true
f0_changes_waveform: true
gradients_finite: true
total_decreased: true
envelope_decreased: true
parameter_budget_pass: true
receptive_field_pass: true
cpu_candidate_pass: true
```

Measured cost:

```text
parameters: 231360
receptive_field_ms: 64.958
mean_seconds_per_step: 1.0
max_seconds_per_step: 1.2294
benchmark_audio_seconds: 1.024
benchmark_inference_seconds_median: 0.6981
benchmark_rtf: 0.6818
```

V5 is therefore both structurally viable and faster than real time in the bounded CPU smoke.

## Bounded resumable trainer candidate

Implemented:

```text
lykenox_voice_engine/training/speech_vocoder_v5_artifact.py
lykenox_voice_engine/training/speech_vocoder_v5_train.py
lykenox_voice_engine/training/speech_vocoder_v5_resume_smoke.py
```

Checkpoint kind:

```text
lykenox_v5_vocoder_training_checkpoint
```

Trainer contract:

```text
v5-bounded-resumable-v1
```

Default future persistent artifact directory:

```text
models/lykenox_identity/training/vocoder_stochastic_glottal_filter_v5/
```

Default persistent schedule, once authorized:

```text
segment_mel_frames: 48
train_items: 118
val_items: 14
max_epochs: 28
warmup_epochs: 4
patience: 6
generator_lr: 2e-4
discriminator_lr: 1e-4
checkpoint_every_updates: 6
time_budget_seconds: 80
```

Stable target-referenced checkpoint selection deliberately does **not** reuse the v4.4 harmonic-exposure proxy because that metric improved without perceptual improvement. V5 selection is:

```text
reconstruction
+ 0.50 * envelope
+ 0.25 * spectral_balance
+ 0.15 * local_spectral_contrast
```

Adversarial and feature-matching losses are training-only after warmup and do not control best-checkpoint selection.

## Current gate — exact resume only

Persistent v5 training is **not authorized yet**. First run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v5_resume_smoke
```

The smoke compares the same four deterministic updates as `4` versus `2 + checkpoint/reload + 2`, with adversarial training active. It requires bit-exact generator, discriminator, both optimizer states, torch RNG, epoch, item offset, global step and run config. It also verifies source-family identity, zero deterministic harmonics, absence of a sinusoidal carrier and hashes the historical v4.2/v4.3/v4.4 best checkpoints before/after.

Required state:

```text
status: pass
trainer_contract_version: v5-bounded-resumable-v1
global_step_exact: true
epoch_exact: true
next_item_offset_exact: true
generator_state_exact: true
discriminator_state_exact: true
generator_optimizer_exact: true
discriminator_optimizer_exact: true
torch_rng_state_exact: true
run_config_exact: true
source_family_exact: true
no_sinusoidal_carrier_exact: true
zero_deterministic_harmonics_exact: true
historical_checkpoints_unchanged: true
temporary_artifacts_removed: true
persistent_v5_training_started: false
next_gate: start_bounded_resumable_v5_persistent_training
```

A pass authorizes persistent v5 training. A failure requires fixing resume semantics before any long run.

## Remaining order

```text
v4.4 path attribution                     [CLOSED: periodic path implicated]
  -> v5 non-sinusoidal architecture smoke [PASS]
  -> exact-resume v5 trainer gate          [CURRENT]
  -> bounded persistent v5 training
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> reference-free text-to-waveform perceptual gate
```

No `/speak`, export or product acceptance is authorized until a vocoder passes full-utterance perceptual acceptance without the radio-mistuned / metallic interference.
