# LYKENOX Vocoder v4.4 — dynamic-filter hybrid corrective gate

## Evidence entering v4.4

V4.3 completed numerical training and passed its structural oracle audit, but full-utterance listening judged it perceptually worse. The failure is described as a radio-mistuned / metallic interference texture attached to the voice rather than a single pure whistle.

A bounded v4.3 carrier ablation found the 8-harmonic equal-RMS variant to be the best of the tested 24/16/12/8-harmonic and voiced-noise variants, but it did not solve the artifact. This established that the original carrier was too coherent/exposed and that reducing carrier complexity alone was insufficient.

## V4.4 corrective principle

Architecture identity:

```text
lykenox_dynamic_filter_hybrid_v4_4
```

V4.4 uses an 8-harmonic anti-aliased periodic branch plus an independent broadband aperiodic branch. Mel selects among three bias-free depthwise filter bases at every residual block and also gates excitation-dependent aperiodic detail. There is no raw source-to-waveform bypass and no mel-only waveform branch.

Structural invariant:

```text
excitation_scale = 0
mel != 0
=> waveform == 0 exactly
```

V4.4 also introduced the target-relative F0-locked harmonic exposure objective:

```text
vocoder-harmonic-exposure-v1
```

which compares harmonic-bin exposure against nearby inter-harmonic energy in the paired real target.

## Architecture and resume gates — PASSED

The bounded CPU architecture smoke passed:

```text
status: pass
architecture: lykenox_dynamic_filter_hybrid_v4_4
harmonics: 8
parameters: 578432
receptive_field_ms: 64.958
benchmark_rtf: 1.7482
no_mel_only_waveform: true
zero_excitation_max_abs: 0.0
gradients_finite: true
total_decreased: true
envelope_decreased: true
harmonic_exposure_decreased: true
```

The exact-resume gate also passed for trainer:

```text
v4-4-bounded-resumable-v1
```

with generator, discriminator, both optimizers, RNG, epoch, item offset, global step and run config reproducing exactly across interruption/reload.

## Persistent training — COMPLETE / CLOSED

Artifact directory:

```text
models/lykenox_identity/training/vocoder_dynamic_filter_hybrid_v4_4/
```

Completed result:

```text
status: pass
stop_reason: max_epochs
persistent_training_complete: true
epochs_completed: 28
global_step: 3304
best_epoch: 27
training_improved: true
envelope_improved: true
harmonic_exposure_improved: true
```

Initial validation:

```text
reconstruction: 3.602273
envelope: 3.558446
spectral_balance: 2.304265
local_spectral_contrast: 0.313305
harmonic_exposure: 0.47218
selection_score: 6.122603
```

Best validation:

```text
reconstruction: 1.276072
envelope: 0.684056
spectral_balance: 0.09466
local_spectral_contrast: 0.286187
harmonic_exposure: 0.358537
selection_score: 1.774328
```

No further v4.4 training is authorized by inertia.

## Full-utterance oracle gate — STRUCTURAL PASS / PERCEPTUAL REJECT

The full-utterance oracle audit compared, on the same three held-out utterances:

```text
reference -> v4.2 oracle -> v4.3 oracle -> v4.4 oracle
```

with target mel + target F0 + target voicing on the teacher frame grid. No gain normalization, training or checkpoint mutation was used.

Structural state:

```text
status: needs_listening
structural_gate_pass: true
persistent_training_complete: true
v4_4_identity_exact: true
v4_3_identity_exact: true
v4_2_identity_exact: true
```

Objective v4.4 improvement counts were:

```text
vs v4.3:
  envelope_loss: 0/3
  reconstruction_loss: 0/3
  spectral_balance_loss: 3/3
  local_spectral_contrast_loss: 0/3
  harmonic_exposure_loss: 3/3

vs v4.2:
  envelope_loss: 0/3
  reconstruction_loss: 0/3
  spectral_balance_loss: 2/3
  local_spectral_contrast_loss: 0/3
  harmonic_exposure_loss: 3/3
```

Listening found **no material perceptual improvement** over v4.2 in the radio-mistuned / metallic interference. Therefore:

```text
v4.4 numerical training: PASS
v4.4 structural oracle gate: PASS
v4.4 perceptual oracle acceptance: REJECTED
v4.4 product acceptance: REJECTED
additional v4.4 training: NOT AUTHORIZED
```

The harmonic-exposure objective and broad spectral-balance improvements are therefore not sufficient proxies for this artifact. A new architecture must not be trained until the remaining v4.4 artifact is attributed to a specific internal path.

## Current gate — v4.4 path attribution, no training

Implemented:

```text
lykenox_voice_engine/training/speech_vocoder_v4_4_path_attribution_ablation.py
```

This bounded audit uses one complete held-out oracle-conditioned utterance, choosing the shortest among the same first three validation candidates to remain safe under the local command limit. It first proves sample-exact reproduction of the trained v4.4 baseline and then renders:

```text
baseline.wav
periodic_only.wav
aperiodic_only.wav
no_block_aperiodic.wav
equal_filter_selector.wav
```

Interpretation:

```text
periodic_only keeps artifact, aperiodic_only does not
  -> periodic path remains dominant

aperiodic_only keeps artifact
  -> aperiodic excitation/filtering is implicated

no_block_aperiodic materially cleans artifact
  -> repeated per-block aperiodic injection is implicated

equal_filter_selector materially cleans artifact
  -> time-varying mel-selected filter-bank modulation is implicated
```

Run only:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_4_path_attribution_ablation
```

No new vocoder training is authorized before this attribution gate is listened to and closed.

## Remaining order

```text
v4.4 architecture smoke                 [PASS]
  -> exact-resume trainer gate          [PASS]
  -> bounded persistent v4.4 training   [PASS / CLOSED]
  -> full-utterance oracle listening    [REJECTED]
  -> v4.4 internal path attribution     [CURRENT]
  -> architecture decision
  -> only then consider a new vocoder candidate
```

Predicted-duration calibration, reference-free text-to-waveform acceptance, `/speak`, export and product packaging remain blocked until a vocoder passes the full-utterance perceptual gate.
