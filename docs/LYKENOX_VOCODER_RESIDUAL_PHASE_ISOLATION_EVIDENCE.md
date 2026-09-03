# LYKENOX residual phase isolation evidence

**Policy:** `LYX-POL-001`  
**Date:** 2026-09-03  
**Renderer:** unchanged fixed minimum-phase renderer  
**Training:** none for this forensic  

## Accepted prior ceilings

The investigation retains these clean oracle references:

- `speech_0021_6cd35984e877_seg_001__identity_roundtrip_ceiling.wav`
- `speech_0022_ba721f6129b9_seg_005__real_residual_resynthesis.wav`

Both establish that the real residual through the fixed minimum-phase path can reconstruct clean,
natural voice.

## Phase/magnitude swap result

The no-training phase/magnitude forensic generated, among other controls:

- candidate magnitude + target residual phase;
- target magnitude + candidate residual phase.

For `speech_0021_6cd35984e877_seg_001`, the owner reported that
`__candidate_mag_target_phase_render.wav` is the output that sounds correct.

The same report measured candidate-vs-target residual phase alignment at approximately `-0.000497`,
while log-magnitude L1 was `0.9904`.

## Interpretation

This is direct perceptual evidence that, for this held-out item, the current predicted residual
magnitude is capable of supporting a clean render when paired with the real residual phase. Therefore
the dominant audible failure of the residual-statistics candidate is isolated to its phase / temporal
coherence path rather than requiring another magnitude model or renderer change.

This does not authorize use of target phase at inference. Target phase is an oracle ceiling only.
The engineering problem is now to recover or generate coherent residual phase without access to the
reference waveform.

## Active gate

Do not train another source model and do not modify the renderer. First test whether coherent phase can
be reconstructed from the already-predicted residual magnitude alone. The authorized diagnostic is:

`scripts/diagnostic_candidate_magnitude_phase_recovery_v1.py`

It uses deterministic 64-iteration Griffin-Lim projection and emits both a target-magnitude control and
a candidate-magnitude phase-recovery render. Human listening against the target-phase ceiling decides
whether phase can be recovered without a learned phase model.
