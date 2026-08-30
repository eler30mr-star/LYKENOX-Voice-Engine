# LYKENOX vocoder V7 — perceptual rejection

**Date:** 2026-08-30  
**Status:** rejected; training disabled  
**Accepted comparison baseline:** v4.2

## Verdict

V7 epoch 1 does not synthesize intelligible speech. In all three held-out full-utterance
oracle files it emits a nearly stationary low-frequency buzz. Words, consonants and formant
motion are absent or unusable. No epoch 2 is authorized.

The failure is not low output level. Raw RMS is close to the references, but that energy is
concentrated in the wrong signal: a frame-grid tone. Overall RMS therefore gave a false sense
of progress and is no longer sufficient as a volume gate.

## Direct waveform evidence

The active audio contract is 24,000 Hz with hop length 256, so the mel-frame rate is:

```
24000 / 256 = 93.75 Hz
```

All three V7 outputs have dominant spectral lines exactly at:

- 93.75 Hz — frame rate;
- 187.5 Hz — second frame-rate harmonic;
- 281.25 Hz — third frame-rate harmonic.

Normalized full-waveform autocorrelation at a 256-sample lag was:

- utterance 1: approximately 0.995;
- utterance 2: approximately 0.997;
- utterance 3: approximately 0.992.

At a 512-sample lag it remained approximately 0.973–0.988. This means the waveform repeats
almost exactly on the mel hop grid.

The three V7 outputs are also abnormally similar to one another despite different texts:
pairwise waveform correlations are approximately 0.53–0.75. The corresponding v4.2
cross-utterance correlations are approximately 0.01–0.04.

## Oracle-report evidence

V7 spectral centroids were only about 184–186 Hz. The references ranged from about 341 to
542 Hz and v4.2 from about 286 to 463 Hz.

Held-out 1–8 kHz presence errors were:

| Utterance | v4.2 | V7 epoch 1 |
|---|---:|---:|
| 1 | 3.23 dB | 8.00 dB |
| 2 | 2.22 dB | 11.66 dB |
| 3 | 2.36 dB | 8.78 dB |

V7 put about 98.4% of the tracked band power into 80–300 Hz on every utterance while
1–3 kHz and 3–8 kHz collapsed.

## Root cause

V7 uses three `ConvTranspose1d` stages with factors `(8, 8, 4)`, whose product is the
256-sample hop. The full-utterance output locks to that upsampling lattice. The observed
93.75/187.5 Hz comb and approximately 0.99 hop autocorrelation are the expected signature of
a severe transposed-convolution checkerboard/frame-grid artifact.

The short architecture smoke was inadequate. It verified shape, gradients and decreasing
crop losses but did not measure:

- hop-period autocorrelation;
- exact frame-rate harmonic concentration;
- cross-input output similarity;
- complete-utterance intelligibility.

## Permanent corrective rules

1. V7 training, architecture smoke and resume smoke are disabled.
2. V7 checkpoints are forensic artifacts only.
3. Every future waveform candidate must pass `vocoder-frame-grid-artifact-v1` before any
   persistent training.
4. Overall RMS cannot grant a volume pass. Useful level must coexist with speech-band energy,
   temporal phonetic variation and full-utterance intelligibility.
5. No future candidate may use learned transposed-convolution upsampling without proving that
   frame-rate comb energy and hop autocorrelation are safely below rejection thresholds.
6. V4.2 remains the only accepted baseline. Future work must start from its intelligible
   behavior and make isolated, reversible changes rather than replacing the decoder wholesale.
7. A candidate that is less intelligible than v4.2 is rejected immediately regardless of
   reconstruction, mel-content or RMS improvements.
