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

This invariant is intentionally tested by the architecture smoke. It prevents the model from learning the v4.2 failure mode where mel and source paths can split waveform authority in a way that leaves periodic source leakage exposed.

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

## Current gate

No persistent v4.3 training is authorized yet.

Run only:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_v4_3_architecture_smoke
```

Required pass conditions include:

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
next_gate: build_bounded_resumable_v4_3_training_candidate
```

The smoke also reports local CPU step time and median inference RTF but does not hide a structural failure behind speed measurements.

## Gate order after a pass

A smoke pass authorizes only construction of an exactly resumable persistent trainer. The remaining order is:

```text
v4.3 architecture smoke
  -> exact-resume trainer gate
  -> bounded persistent v4.3 training
  -> full-utterance oracle listening acceptance
  -> predicted-duration calibration
  -> full reference-free text-to-waveform perceptual gate
```

No `/speak` integration or product-runtime acceptance is authorized before the full-utterance oracle waveform gate passes without the characteristic metallic/chillido artifact.
