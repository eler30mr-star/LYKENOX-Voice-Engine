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

The carrier is no longer added to a separate mel waveform path. Instead, every audible sample must originate from the carrier and pass through a mel-controlled nonlinear filter.

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

## Richer carrier, stronger filtering obligation

V4.1/v4.2 used eight explicit harmonics. Removing harmonics degraded useful voice, so v4.3 does not solve the residual artifact by starving the excitation. The candidate uses 24 deterministic harmonics with `1/sqrt(h)` weighting, total harmonic RMS normalization and a smooth anti-alias guard.

The richer carrier is not a direct waveform. It is required to pass through the mel-conditioned filter before projection.

## New target-relative local spectral-contrast loss

Broad-band spectral balance cannot directly detect narrow metallic peaks. V4.3 therefore adds a training-only local log-STFT contrast objective:

```text
log magnitude spectrum
  - local frequency-smoothed log magnitude
  -> local spectral contrast
  -> compare prediction against paired real target
```

Because the comparison is target-relative, natural harmonic speech structure is not globally penalized. The objective targets excess narrow peak/notch structure relative to the real recording.

Loss identity:

```text
vocoder-local-spectral-contrast-v1
```

## Architecture smoke — PASSED

The bounded real-data CPU smoke passed locally:

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

Bounded optimization also reduced the composite, envelope and local-contrast objectives:

```text
probe_before.total: 6.311642
probe_after.total: 2.628071
probe_before.log_mel_envelope: 3.790301
probe_after.log_mel_envelope: 1.280633
probe_before.local_spectral_contrast: 0.323659
probe_after.local_spectral_contrast: 0.302628
```

This closes the architecture gate. It does not authorize persistent training by itself.

## Persistent trainer candidate

The separate v4.3 checkpoint/trainer contract is now implemented in:

```text
lykenox_voice_engine/training/speech_vocoder_v4_3_artifact.py
lykenox_voice_engine/training/speech_vocoder_v4_3_train.py
```

Trainer identity:

```text
v4-3-bounded-resumable-v1
```

Production artifact directory, once authorized:

```text
models/lykenox_identity/training/vocoder_mel_filtered_carrier_v4_3/
```

The stable target-referenced validation selection score is:

```text
reconstruction
+ 0.50 * envelope
+ 0.25 * spectral_balance
+ 0.20 * local_spectral_contrast
```

Adversarial and feature-matching terms may train the generator after warmup, but they are intentionally excluded from checkpoint selection.

## Current gate: exact resume

**Do not start persistent v4.3 training yet.** First prove bit-exact interruption/resume semantics.

Run only:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_3_resume_smoke
```

The smoke executes the same four deterministic updates as `4` versus `2 + checkpoint/reload + 2` in temporary directories and requires equality of generator, discriminator, both optimizers, RNG, epoch, item offset and global step. It also verifies that the historical v4.2 best checkpoint is unchanged.

Required result:

```text
status: pass
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
persistent_v4_3_training_started: false
next_gate: start_bounded_resumable_v4_3_persistent_training
```

Only after that exact-resume gate passes is the long v4.3 run authorized.

## Remaining gate order

```text
v4.3 architecture smoke          [PASS]
  -> exact-resume trainer gate   [CURRENT]
  -> bounded persistent v4.3 training
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> full reference-free text-to-waveform perceptual gate
```

No `/speak` integration or product-runtime acceptance is authorized before the full-utterance oracle waveform gate passes without the characteristic metallic/chillido artifact.
