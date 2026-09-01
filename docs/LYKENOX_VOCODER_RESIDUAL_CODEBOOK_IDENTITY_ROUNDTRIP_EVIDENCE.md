# LYKENOX residual-codebook identity roundtrip — perceptual evidence

**Date:** 2026-09-01  
**Policy:** LYX-POL-001  
**Status:** PASS for representation/OLA isolation; no selector training authorized

## Purpose

This diagnostic isolates the residual-codebook representation itself from codeword substitution.
The exact owned held-out real residual demonstrated by the positive Step-3f oracle is analyzed with
the codebook's 512-sample sqrt-Hann windows at 256-sample hop, then the exact same analysis vectors
are overlap-added back with no codeword replacement and no oracle gain. The reconstructed residual
is finally passed through the unchanged minimum-phase filter path.

## Owner-reported listening result

The owner reports the following on complete held-out audio:

- `__identity_roundtrip_residual.wav` sounds poor/noisy and is not speech-like by itself;
- `__identity_roundtrip_resynthesis.wav` sounds correct and matches the original voice audio.

The pre-filter residual is not required to sound like speech. The decisive result is the final
resynthesis: using the exact real-residual vectors after the 512/256 analysis/synthesis roundtrip
returns the clean original voice character.

## Conclusion

The following components are exculpated for the gangoso defect observed in residual-codebook oracle
V2:

1. the 512-sample codevector window length;
2. the 256-sample codevector hop;
3. the paired sqrt-Hann overlap-add representation;
4. the frozen minimum-phase filter path when given the correct residual sequence.

Therefore the V2 failure is localized to **codeword substitution/selection and sequence coherence**.
Replacing each real-residual window independently with a different train codeword does not preserve
the excitation trajectory required by the clean Step-3f ceiling.

The fact that the residual file itself sounds like noise/chill rather than speech is not a rejection
criterion. A source residual can be perceptually non-speech-like while still being the correct
excitation for the frozen filter.

## Engineering gate

No selector/model training is authorized by this result. The next diagnostic must remain oracle-only
and owned-data-only. It must preserve temporal coherence across selected train codewords and must not
introduce per-window arbitrary polarity flips. Only after a coherent codebook sequence demonstrates
clear held-out audible capacity may a separate LYKENOX-owned selector be considered under a new
explicit training gate.

No production renderer change, model training, optimizer, checkpoint, post-hoc gain normalization,
EQ, denoise, enhancement, third-party voice component, or remote inference is authorized by this
evidence.
