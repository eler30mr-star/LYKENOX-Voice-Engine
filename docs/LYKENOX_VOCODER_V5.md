# LYKENOX Vocoder v5 — stochastic glottal filter

## Purpose

V5 was the first architectural break after the v4.x path-attribution result implicated the explicit periodic carrier in the radio-mistuned / metallic artifact. It removed the sinusoidal/harmonic bank and used stochastic glottal-pulse/noise excitation instead.

Architecture:

```text
lykenox_stochastic_glottal_filter_v5
```

Source family:

```text
stochastic_glottal_pulse_noise
```

Hard identity:

```text
explicit_sinusoidal_carrier: false
deterministic_harmonics: 0
```

## Structural and training gates

Architecture smoke: **PASS**.

```text
parameters: 231360
receptive_field_ms: 64.958
benchmark_rtf: 0.6818
```

Exact-resume trainer gate: **PASS**.

Trainer:

```text
v5-bounded-resumable-v1
```

Persistent training: **PASS / CLOSED**.

```text
status: pass
stop_reason: max_epochs
persistent_training_complete: true
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

No more v5 training is authorized.

## Full-utterance oracle gate — REJECTED

The oracle audit passed structurally and preserved checkpoint identity, but listening rejected v5. The supplied full-utterance v5 outputs were judged **worse**, with speech described as noisy/gangoso/nasal and subjectively too weak.

Objective v5 improvement counts were also poor:

```text
vs v4.4:
  envelope_loss: 0/3
  reconstruction_loss: 0/3
  spectral_balance_loss: 0/3
  local_spectral_contrast_loss: 1/3
  harmonic_exposure_diagnostic_loss: 0/3

vs v4.2:
  envelope_loss: 0/3
  reconstruction_loss: 0/3
  spectral_balance_loss: 0/3
  local_spectral_contrast_loss: 0/3
  harmonic_exposure_diagnostic_loss: 1/3
```

## Root-cause conclusion

Removing the sinusoidal bank did not solve the product problem because v5 still forced all audible voiced speech through an explicit stochastic source. The `zero excitation => zero waveform` invariant made that source unavoidable.

The perceived low volume is also not primarily a simple global-gain problem. Analysis of the three supplied reference/v5 pairs showed global RMS remaining close to reference while spectral centroid and 1–8 kHz presence dropped materially. The result can therefore sound quieter and less intelligible even when RMS is similar.

Closed decision:

```text
v5 numerical training: PASS
v5 structural oracle: PASS
v5 perceptual acceptance: REJECTED
v5 product acceptance: REJECTED
additional v5 training: NOT AUTHORIZED
v5.1 source/noise tuning: NOT AUTHORIZED
```

The corrective architecture is documented in:

```text
docs/LYKENOX_VOCODER_V6.md
```

V6 removes the explicit voiced source path itself and adds target-relative level and spectral-presence objectives. Persistent v6 training is not authorized until its bounded architecture smoke passes.
