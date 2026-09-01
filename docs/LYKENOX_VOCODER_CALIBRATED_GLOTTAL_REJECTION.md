# LYKENOX calibrated glottal excitation — perceptual rejection

**Date:** 2026-09-01  
**Status:** rejected for production integration  
**Policy:** LYX-POL-001  
**Production renderer modified:** no  
**Training executed:** no neural training in this candidate path

## Verdict

The calibrated Rosenberg glottal-pulse + measured multi-band aperiodicity candidate remains
perceptually rejected. The owner reports that the complete held-out oracle audio is still
characteristically gangoso/rough despite calibration from owned voice data.

This is not treated as a calibration-tuning failure. The local run reported 97,168 measured
pitch-synchronous cycles feeding the owned calibration artifacts, yet the characteristic defect
remained. That is sufficient perceptual evidence to stop iterating constants or pulse-shape
parameters inside this parametric synthetic-excitation family.

The 97,168-cycle count and listening result are owner-reported local-run evidence; they are
recorded here for traceability and are not claimed as independently re-executed by GitHub CI.

## Evidence carried forward

The earlier real-residual analysis/resynthesis diagnostic remains the strongest positive result:
when the synthetic excitation is removed and the owned real residual is passed through the same
order-64 minimum-phase envelope/filter path, the owner reports clean, natural output matching the
original voice recording.

Therefore:

- the minimum-phase envelope/filter path has demonstrated a clean oracle ceiling;
- crossfade changes did not remove the defect;
- replacing hash noise with Gaussian noise did not remove the defect;
- lowering cepstral order did not remove the defect;
- splitting periodic/aperiodic excitation by frequency improved the sound only partially;
- calibrating a Rosenberg pulse and four-band aperiodicity from owned residual measurements still
  did not remove the defect.

## Root-cause interpretation

The dominant remaining failure is the *parametric synthetic source assumption itself*: a compact
pulse/noise generator is not reproducing the fine residual structure required by this voice, even
when its coarse temporal, spectral-tilt and band-aperiodicity parameters are measured from owned
recordings.

No further production work is authorized on generic or calibrated pulse + noise tuning without a
new piece of evidence that changes this conclusion.

## Why pure CELP codebook search is not the direct TTS solution

A CELP-style codebook remains potentially useful as a representation of owned residual shapes, but
classic analysis-by-synthesis codebook search assumes access to a target speech/residual signal at
the encoder. The LYKENOX TTS inference path does not have the target residual waveform available,
so a codebook by itself cannot determine which excitation entry to use for unseen text.

Any codebook approach therefore still requires an owned inference-time selector/predictor from the
available conditioning to code index, gain and/or residual coefficients. For that reason, the next
active engineering path is a narrowly scoped owned residual/excitation predictor rather than a
standalone CELP search loop.

## Authorized next direction

The next candidate must predict only the excitation/residual detail while keeping the already-proven
minimum-phase envelope/filter path fixed.

Required constraints:

1. CPU-only.
2. LYKENOX implementation and LYKENOX-trained weights only.
3. No third-party pretrained model, checkpoint, speaker encoder, vocoder, codec or remote API.
4. Targets are owned real residuals extracted with the same method proven in the real-residual
   diagnostic.
5. The predictor must not learn or bypass the spectral-envelope/filter path.
6. Full held-out utterance listening remains the acceptance authority.
7. Metrics may reject but cannot accept product quality.
8. No post-hoc gain normalization, EQ, denoise or duration modification may hide failures.

A codebook learned only from owned residuals may be used internally to reduce the prediction problem
(e.g. predict an owned residual code/index plus gain), but the inference-time selector must itself be
LYKENOX-owned and trained only on owned data.

## Next action

Design the smallest CPU-trainable residual/excitation target and predictor that can be validated
against the real-residual oracle ceiling before reopening bounded end-to-end vocoder training.
