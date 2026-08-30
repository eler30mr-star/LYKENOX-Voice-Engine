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

```text
status: pass
architecture: lykenox_stochastic_glottal_filter_v5
source_family: stochastic_glottal_pulse_noise
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
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

Measured local CPU cost:

```text
parameters: 231360
receptive_field_ms: 64.958
mean_seconds_per_step: 1.0
max_seconds_per_step: 1.2294
benchmark_audio_seconds: 1.024
benchmark_inference_seconds_median: 0.6981
benchmark_rtf: 0.6818
```

V5 is structurally viable and faster than real time in the bounded CPU smoke.

## Exact-resume gate — PASSED

Trainer contract:

```text
v5-bounded-resumable-v1
```

The exact-resume smoke compared the same four deterministic updates as `4` versus `2 + checkpoint/reload + 2` with adversarial training active. Generator, discriminator, both optimizers, torch RNG, epoch, item offset, global step and run config matched exactly. Source-family identity remained exact, deterministic harmonics remained zero, and the historical v4.2/v4.3/v4.4 checkpoints remained unchanged.

## Persistent training — COMPLETE / NUMERICAL PASS

Artifact directory:

```text
models/lykenox_identity/training/vocoder_stochastic_glottal_filter_v5/
```

Closed result:

```text
status: pass
stop_reason: max_epochs
persistent_training_complete: true
architecture: lykenox_stochastic_glottal_filter_v5
source_family: stochastic_glottal_pulse_noise
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
trainer_contract_version: v5-bounded-resumable-v1
epochs_completed: 28
global_step: 3304
best_epoch: 27
training_improved: true
envelope_improved: true
historical_checkpoints_mutated: false
```

Initial validation:

```text
reconstruction: 3.961914
envelope: 4.125153
spectral_balance: 2.155811
local_spectral_contrast: 0.310693
selection_score: 6.610047
```

Best validation:

```text
reconstruction: 1.290753
envelope: 0.701332
spectral_balance: 0.070128
local_spectral_contrast: 0.302039
selection_score: 1.704257
```

Checkpoint selection deliberately excludes the v4.4 harmonic-exposure proxy:

```text
reconstruction
+ 0.50 * envelope
+ 0.25 * spectral_balance
+ 0.15 * local_spectral_contrast
```

No further v5 training is authorized by inertia. Numerical improvement does not grant perceptual or product acceptance.

## Current gate — full-utterance oracle listening acceptance

Implemented:

```text
lykenox_voice_engine/training/speech_vocoder_v5_full_utterance_oracle_acceptance.py
```

The audit uses the same three held-out validation utterances and the same teacher-frame oracle conditioning. Listening order:

```text
reference -> v4.2 oracle -> v4.4 oracle -> v5 oracle
```

V4.2 is retained as the stronger historical baseline and v4.4 is retained as the last rejected architecture before the non-sinusoidal break. All candidates receive identical target mel + target F0 + target voicing. No gain normalization, training or checkpoint mutation is allowed.

The audit reports reconstruction, envelope, spectral balance and local spectral contrast. Harmonic-exposure is included only as a diagnostic field because v4.4 demonstrated that it cannot grant perceptual acceptance.

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v5_full_utterance_oracle_acceptance
```

Expected structural state before listening:

```text
status: needs_listening
structural_gate_pass: true
persistent_training_complete: true
v5_identity_exact: true
v4_4_identity_exact: true
v4_2_identity_exact: true
checkpoints_unchanged: true
full_utterance_perceptual_acceptance: false
next_gate: listen_v5_full_utterance_oracle_sets_and_accept_or_revise_vocoder
```

Perceptual acceptance requires the radio-mistuned / metallic interference to be absent or materially resolved on all three full held-out utterances while preserving useful voice body, intelligibility, consonants/formants and level. Improvement versus rejected v4.4 alone is insufficient: v5 must also be perceptually preferable to v4.2.

## Remaining order

```text
v4.4 path attribution                     [CLOSED: periodic path implicated]
  -> v5 non-sinusoidal architecture smoke [PASS]
  -> exact-resume v5 trainer gate          [PASS]
  -> bounded persistent v5 training        [PASS / CLOSED]
  -> full-utterance oracle listening       [CURRENT]
  -> predicted-duration calibration
  -> reference-free text-to-waveform perceptual gate
```

No `/speak`, export or product acceptance is authorized until a vocoder passes full-utterance perceptual acceptance without the radio-mistuned / metallic interference.
