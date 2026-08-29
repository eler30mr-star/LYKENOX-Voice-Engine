# LYKENOX Vocoder v4.4 — dynamic-filter hybrid corrective gate

## Evidence entering v4.4

V4.3 completed numerical training and passed its structural oracle audit, but listening judged the full-utterance result perceptually worse. The failure is better described as a radio-mistuned / metallic carrier interference texture attached to the voice rather than a single pure whistle.

A bounded post-training carrier ablation then compared the exact 24-harmonic v4.3 baseline against 16, 12 and 8 active harmonics at approximately equal harmonic RMS, plus higher voiced-noise floors. Listening found the 8-harmonic variant to be the best of the tested candidates, but not solved.

This established two facts:

1. the 24-harmonic deterministic carrier was too coherent/exposed for the trained v4.3 filter;
2. reducing carrier complexity alone was insufficient, so v4.3's scalar multiplicative-only mel control was also too restrictive.

V4.3 remains rejected for product use. No further v4.3 training is authorized.

## V4.4 corrective principle

V4.4 keeps an eight-harmonic anti-aliased carrier and introduces an independent broadband aperiodic excitation state. Mel no longer controls the carrier through only scalar gains. Instead, each residual block contains multiple bias-free convolutional filter bases and mel predicts the time-varying mixture among those filters.

Architecture identity:

```text
lykenox_dynamic_filter_hybrid_v4_4
```

Core structure:

```text
8-harmonic periodic excitation + voiced/log-F0
  -> bias-free periodic stem

broadband deterministic aperiodic excitation
  -> bias-free aperiodic stem

periodic + aperiodic states
  -> bias-free initial mix
  -> 8 dilated dynamic filter-bank blocks
     - 3 bias-free depthwise filter bases per block
     - mel selects filter-basis mixture
     - mel gates excitation-dependent aperiodic detail
  -> bias-free waveform projection
  -> fixed 30 Hz high-pass FIR
  -> waveform
```

There is no raw periodic source bypass and no mel-only waveform branch.

## Structural invariant

Every audible path is excitation-dependent:

```text
excitation_scale = 0
mel != 0
=> waveform == 0 exactly
```

This preserves the anti-shortcut property learned from v4.2/v4.3 while giving mel substantially richer spectral-filter authority than v4.3's scalar multiplicative gains.

## F0-locked harmonic exposure loss

V4.3 proved that log-mel envelope and broad-band balance can improve while the radio-like carrier texture becomes perceptually worse. V4.4 therefore adds a training-only target-relative objective:

```text
harmonic bin level
- nearby inter-harmonic level
= harmonic exposure
```

For harmonics 1 through 8, predicted harmonic exposure is compared against the paired real target at the target F0 trajectory. Natural harmonic speech structure is therefore preserved rather than globally suppressed.

Loss identity:

```text
vocoder-harmonic-exposure-v1
```

## Architecture smoke — PASSED

The bounded CPU architecture smoke passed locally:

```text
status: pass
architecture: lykenox_dynamic_filter_hybrid_v4_4
harmonics: 8
persistent_training_started: false
v4_3_checkpoint_mutated: false
exact_length_contract: true
structural_finite: true
no_mel_only_waveform: true
zero_excitation_max_abs: 0.0
gradients_finite: true
total_decreased: true
envelope_decreased: true
harmonic_exposure_decreased: true
parameter_budget_pass: true
receptive_field_pass: true
```

Measured local CPU cost:

```text
parameters: 578432
receptive_field_ms: 64.958
mean_seconds_per_step: 2.9666
max_seconds_per_step: 3.9201
benchmark_audio_seconds: 1.024
benchmark_inference_seconds_median: 1.7901
benchmark_rtf: 1.7482
```

The architecture is structurally viable as a corrective candidate, but it is currently slower than real time on the local CPU. Runtime optimization remains secondary to perceptual acceptance.

## Exact-resume gate — PASSED

Trainer identity:

```text
v4-4-bounded-resumable-v1
```

The exact-resume CPU smoke passed after comparing the same four deterministic updates as `4` versus `2 + checkpoint/reload + 2` with adversarial training active. Generator, discriminator, both optimizers, RNG, epoch, item offset, global step and run config all matched exactly. The historical v4.3 checkpoint remained unchanged.

## Persistent training — COMPLETE / NUMERICAL PASS

Persistent artifact directory:

```text
models/lykenox_identity/training/vocoder_dynamic_filter_hybrid_v4_4/
```

The bounded resumable run completed and is closed. No further v4.4 training is authorized by inertia.

```text
status: pass
stop_reason: max_epochs
persistent_training_complete: true
architecture: lykenox_dynamic_filter_hybrid_v4_4
trainer_contract_version: v4-4-bounded-resumable-v1
epochs_completed: 28
global_step: 3304
best_epoch: 27
training_improved: true
envelope_improved: true
harmonic_exposure_improved: true
v4_3_checkpoint_mutated: false
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

The stable target-referenced checkpoint selection score was:

```text
reconstruction
+ 0.50 * envelope
+ 0.25 * spectral_balance
+ 0.15 * local_spectral_contrast
+ 0.25 * harmonic_exposure
```

This numerical pass does **not** grant perceptual or product acceptance.

## Current gate — full-utterance oracle listening

Implemented audit:

```text
lykenox_voice_engine/training/speech_vocoder_v4_4_full_utterance_oracle_acceptance.py
```

The audit uses the same three fixed held-out validation utterances and preserves both historical baselines because v4.3 was perceptually worse than v4.2:

```text
reference -> v4.2 oracle -> v4.3 oracle -> v4.4 oracle
```

All synthesized candidates receive the same target mel + target F0 + target voicing on the teacher frame grid. Reference waveform and target pitch are audit-only inputs. No training, checkpoint mutation, gain normalization, source speaker or product-time reference audio is introduced.

The report compares:

```text
reconstruction
envelope
spectral_balance
local_spectral_contrast
harmonic_exposure over harmonics 1..8
RMS / spectral-band diagnostics
```

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_4_full_utterance_oracle_acceptance
```

Expected pre-listening state:

```text
status: needs_listening
structural_gate_pass: true
persistent_training_complete: true
full_utterance_perceptual_acceptance: false
next_gate: listen_v4_4_full_utterance_oracle_sets_and_accept_or_revise_vocoder
```

Perceptual acceptance requires the radio-mistuned / metallic carrier interference to be absent or materially resolved on all three complete held-out utterances, while preserving intelligibility, consonants/formants and usable level. Improvement versus rejected v4.3 alone is insufficient: v4.4 must also be perceptually no worse than the stronger historical v4.2 baseline.

## Remaining order

```text
v4.4 architecture smoke                [PASS]
  -> exact-resume trainer gate         [PASS]
  -> bounded persistent v4.4 training  [PASS / CLOSED]
  -> full-utterance oracle listening acceptance  [CURRENT]
  -> predicted-duration calibration
  -> reference-free text-to-waveform perceptual gate
```

No `/speak`, export or product acceptance is authorized before full-utterance oracle listening demonstrates that the radio-mistuned / metallic carrier interference is materially resolved without sacrificing intelligibility or useful level.
