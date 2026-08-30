# LYKENOX vocoder V7 — source-free mel-latent decoder

**Status:** perceptually rejected; training disabled  
**Rejection date:** 2026-08-30  
**Comparison baseline:** v4.2  
**Detailed evidence:** `docs/LYKENOX_VOCODER_V7_REJECTION.md`

## Final verdict

V7 failed its mandatory epoch-1 full-utterance listening gate. All three held-out outputs
collapsed to a nearly stationary frame-grid buzz instead of intelligible speech. No epoch 2
is authorized and all V7 checkpoints are forensic artifacts only.

The failure is tied to the learned transposed-convolution upsampling lattice. With a 24 kHz
sample rate and 256-sample hop, the generated waveforms contain dominant lines at 93.75 Hz
and 187.5 Hz and approximately 0.99 normalized autocorrelation at the hop. Raw RMS was close
to the reference only because the grid tone carried the energy; it did not represent useful
speech volume.

## Historical goal

V7 was intended to recover intelligible, natural speech without the hidden periodic shortcuts
that made V6 sound like a whine/buzz with gangoso coloration. It did remove explicit source
construction, but the learned upsampler created a different periodic shortcut.

## Architecture contract

V7 consumes mel, F0, and voicing at mel-frame rate. F0 and voicing are fused with mel into a
learned frame latent. The waveform decoder receives only that learned latent through learned
upsampling and multi-receptive-field residual refinement.

The following remained absent:

- accumulated sample phase;
- sine/cosine carrier generation;
- harmonic banks;
- pulse/aperture excitation;
- deterministic or stochastic noise excitation;
- raw source bypasses;
- sample-rate F0/voicing control tensors;
- local/global unit-RMS normalization of an arbitrary waveform shape;
- a separate level-rescue branch.

Those properties were not sufficient. `ConvTranspose1d` upsampling with total ratio 256
introduced a severe frame-grid/checkerboard artifact.

## Content objective

V7 used `vocoder-v7-mel-content-v1`, a differentiable waveform-to-log-mel consistency loss
against the conditioning mel. It tracked log-mel level, centered spectral shape, spectral and
temporal deltas, and temporal acceleration.

The loss decreased, but it did not prevent the decoder from converging first to a strong
hop-locked tone. Improvement from random initialization was therefore not an acceptance gate.
Future candidates must be compared directly with v4.2 and must pass the waveform-level
`vocoder-frame-grid-artifact-v1` detector before persistent training.

## Gate outcome

1. Unit contract tests: passed historically.
2. Short-crop architecture smoke: passed historically but is now invalidated.
3. Exact-resume smoke: passed historically; resume correctness does not imply audio quality.
4. First bounded epoch: completed at global step 118.
5. Full-utterance A/B against v4.2: failed decisively.
6. Epoch 2: permanently blocked.

## Product invariants

- Predicted duration was unchanged.
- No post-hoc gain or normalization was used.
- No post-hoc EQ was used.
- No post-hoc denoising was used.
- V4.2 remains the accepted comparison baseline.
- Objective metrics may reject a candidate but cannot grant perceptual acceptance.
- Overall RMS cannot grant a volume pass unless intelligible speech energy is preserved.
