# LYKENOX vocoder V7 — source-free mel-latent decoder

**Status:** architecture candidate only  
**Comparison baseline:** v4.2  
**V6 status:** perceptually rejected and training-disabled

## Goal

V7 exists to recover intelligible, natural speech without the hidden periodic shortcuts that
made V6 sound like a whine/buzz with gangoso coloration. It is not a continuation of V6 and
must never load V6 generator or optimizer state.

## Architecture contract

V7 consumes mel, F0, and voicing at mel-frame rate. F0 and voicing are fused with mel into a
learned frame latent. The waveform decoder receives only that learned latent through learned
upsampling and multi-receptive-field residual refinement.

The following are forbidden:

- accumulated sample phase;
- sine/cosine carrier generation;
- harmonic banks;
- pulse/aperture excitation;
- deterministic or stochastic noise excitation;
- raw source bypasses;
- sample-rate F0/voicing control tensors;
- local/global unit-RMS normalization of an arbitrary waveform shape;
- a separate level-rescue branch that can make an uninformative waveform audible.

The V7 public contract therefore requires:

- `source_free = true`
- `sample_phase_conditioning = false`
- `sample_rate_pitch_features = false`
- `pitch_conditioning_scope = frame_latent_only`
- `deterministic_noise_conditioning = false`
- `local_unit_rms_shape_normalization = false`
- `global_unit_rms_shape_normalization = false`
- `level_rescue_branch = false`
- `perceptually_accepted = false` until listening says otherwise

## Content objective

V6 demonstrated that band fractions and RMS are not intelligibility metrics. V7 adds
`vocoder-v7-mel-content-v1`, a differentiable waveform -> log-mel consistency loss against the
conditioning mel itself. It protects:

1. log-mel level;
2. frame-local centered spectral shape;
3. spectral deltas across mel bins;
4. temporal deltas;
5. temporal acceleration.

This is target-relative and does not impose a generic EQ curve.

## Gate order

1. Unit contract tests.
2. Bounded real-data architecture smoke. No persistent checkpoint may be created.
3. Only if the smoke passes: build an isolated bounded/exact-resume V7 trainer.
4. Exact-resume smoke must pass before persistent training.
5. Complete exactly one bounded epoch.
6. Immediately generate complete held-out oracle utterances and A/B against v4.2.
7. If words are less intelligible than v4.2, or whine/gangoso/metallic coloration is worse,
   V7 is rejected immediately regardless of objective metrics.
8. Only perceptual success can authorize additional epochs.

Objective metrics may reject a candidate but cannot grant perceptual acceptance.

## Product invariants

- Predicted duration is unchanged.
- No post-hoc gain or normalization may hide low model output.
- No post-hoc EQ may hide spectral failure.
- No post-hoc denoising may hide waveform failure.
- V4.2 remains the accepted comparison baseline until a successor audibly beats it.
