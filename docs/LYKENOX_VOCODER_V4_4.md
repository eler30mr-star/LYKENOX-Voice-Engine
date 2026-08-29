# LYKENOX Vocoder v4.4 — dynamic-filter hybrid corrective gate

## Evidence entering v4.4

V4.3 completed numerical training and passed its structural oracle audit, but listening judged the full-utterance result perceptually worse. The failure is better described as a radio-mistuned / metallic carrier interference texture attached to the voice rather than a single pure whistle.

A bounded post-training carrier ablation then compared the exact 24-harmonic v4.3 baseline against 16, 12 and 8 active harmonics at approximately equal harmonic RMS, plus higher voiced-noise floors. Listening found the 8-harmonic variant to be the best of the tested candidates, but not solved.

This establishes two facts:

1. the 24-harmonic deterministic carrier is too coherent/exposed for the trained v4.3 filter;
2. reducing carrier complexity alone is insufficient, so v4.3's scalar multiplicative-only mel control is also too restrictive.

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

Every audible path is excitation-dependent. The architecture smoke explicitly requires:

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

The architecture is therefore structurally viable as a corrective candidate, but it is currently slower than real time on the local CPU. Runtime optimization is not a release blocker yet because perceptual acceptance comes first.

## Bounded resumable trainer candidate

Implemented files:

```text
lykenox_voice_engine/training/speech_vocoder_v4_4_artifact.py
lykenox_voice_engine/training/speech_vocoder_v4_4_train.py
lykenox_voice_engine/training/speech_vocoder_v4_4_resume_smoke.py
```

Trainer identity:

```text
v4-4-bounded-resumable-v1
```

Default persistent artifact directory, once authorized:

```text
models/lykenox_identity/training/vocoder_dynamic_filter_hybrid_v4_4/
```

The stable target-referenced validation selection score is:

```text
reconstruction
+ 0.50 * envelope
+ 0.25 * spectral_balance
+ 0.15 * local_spectral_contrast
+ 0.25 * harmonic_exposure
```

Adversarial and feature-matching terms may train the generator after warmup, but they are excluded from checkpoint selection.

Because v4.4 is materially slower than v4.3, the default persistent crop is 48 mel frames and the default invocation budget is 82 seconds, with checkpoint and validation reserves chosen to stay below the local two-minute command ceiling. Checkpoints are written every four updates and can resume exactly.

## Current gate — exact resume

Persistent v4.4 training is **not authorized yet**. First prove exact interruption/resume semantics.

Run only:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_4_resume_smoke
```

The smoke executes the same four deterministic updates as `4` versus `2 + checkpoint/reload + 2`, with adversarial training active, and requires equality of generator, discriminator, both optimizers, RNG, epoch, item offset, global step and run config. It also hashes the historical v4.3 best checkpoint before and after.

Required result:

```text
status: pass
trainer_contract_version: v4-4-bounded-resumable-v1
global_step_exact: true
epoch_exact: true
next_item_offset_exact: true
generator_state_exact: true
discriminator_state_exact: true
generator_optimizer_exact: true
discriminator_optimizer_exact: true
torch_rng_state_exact: true
run_config_exact: true
v4_3_checkpoint_unchanged: true
temporary_artifacts_removed: true
persistent_v4_4_training_started: false
next_gate: start_bounded_resumable_v4_4_persistent_training
```

Only after that pass is the persistent v4.4 run authorized.

## Remaining order

```text
v4.4 architecture smoke          [PASS]
  -> exact-resume trainer gate   [CURRENT]
  -> bounded persistent v4.4 training
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> reference-free text-to-waveform perceptual gate
```

No `/speak`, export or product acceptance is authorized before full-utterance oracle listening demonstrates that the radio-mistuned / metallic carrier interference is materially resolved without sacrificing intelligibility or useful level.
