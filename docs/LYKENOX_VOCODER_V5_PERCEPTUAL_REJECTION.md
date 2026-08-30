# LYKENOX Vocoder v5 — perceptual rejection and root-cause correction

## Decision

The trained v5 checkpoint is numerically complete but perceptually rejected.

```text
architecture: lykenox_stochastic_glottal_filter_v5
persistent training: PASS / CLOSED
full-utterance structural oracle gate: PASS
perceptual acceptance: REJECTED
product acceptance: REJECTED
additional v5 training: NOT AUTHORIZED
```

Listening describes v5 speech as worse than the historical baseline and notably **gangosa / muffled / nasal**, with the prior radio-like interference problem not replaced by a clean natural voice.

## Evidence from the uploaded oracle WAVs

The same held-out oracle cases were inspected against their real references. V5 consistently collapses spectral energy toward the low band and under-represents the mid/high speech region that carries formant definition and consonant clarity.

Representative whole-utterance spectral centroids:

```text
case 01: reference 384.97 Hz -> v5 287.25 Hz
case 02: reference 522.90 Hz -> v5 378.11 Hz
case 03: reference 339.37 Hz -> v5 289.39 Hz
```

V5 also shifts a larger share of energy below 300 Hz while reducing approximately 1-8 kHz speech detail. Across the three cases, 1-3 kHz energy is roughly 2-3 dB below the paired reference and 3-8 kHz is also reduced. This is consistent with the perceived dark/gangosa coloration and weaker consonant/formant definition.

## Architectural cause

V5 removed the explicit sinusoidal carrier, but it retained a more fundamental source constraint:

```text
broadband deterministic noise
  -> stochastic glottal pulse bursts / voiced noise floor / unvoiced noise
  -> bias-free excitation-dependent filtering
  -> waveform
```

The model contract also enforced:

```text
zero excitation => zero waveform
```

Therefore every audible voiced component must be reconstructed from a noise-derived excitation. Mel and F0 can only gate/select filters around that excitation; they cannot create the waveform directly.

This is now considered over-constrained. The stochastic source is not merely an optional noise layer that can be filtered away: it is the audible substrate of the entire generated voice. Reducing or suppressing it also removes the speech itself.

## Corrective decision

Do not create another source/filter tuning pass. The next architecture must remove **audible explicit excitation** entirely.

Required next-family contract:

```text
target mel + log-F0 + voicing
  -> learned frame encoder
  -> learned anti-aliased upsampling
  -> learned residual waveform generator
  -> waveform
```

F0 and voicing remain conditioning features only. They must not generate a carrier, pulse train, harmonic bank, or broadband-noise source.

The previous `zero excitation => zero waveform` invariant is intentionally retired because there will be no explicit excitation input. The replacement invariants are:

```text
no reference audio at inference
no source speaker/singer
no voice conversion
no sinusoidal carrier
no deterministic harmonic bank
no stochastic voiced-noise source
exact frame-to-waveform length
finite waveform
mel changes waveform
F0 changes voiced waveform as conditioning
voicing changes voiced/unvoiced behavior
```

Learned upsampling should use interpolation + convolution/residual filtering rather than transposed-convolution shortcuts, to reduce checkerboard/tonal artifacts while keeping exact 256-sample expansion per mel frame.

## Gate policy

No persistent training is authorized now. First replace the architecture and prove only its structural/CPU contract. Long training is prohibited until the direct neural waveform path exists and the source-derived failure mode is structurally impossible.
