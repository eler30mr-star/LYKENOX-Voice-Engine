# LYKENOX Vocoder — Temporal Phase Increment Listening Evidence

**Policy:** `LYX-POL-001`  
**Date:** 2026-09-03  
**Diagnostic:** `scripts/diagnostic_candidate_magnitude_temporal_phase_increment_v1.py`  
**Training executed:** no  
**Renderer modified:** no  
**Post-processing:** none

## Fixed positive references

The investigation retains these clean oracle references as authoritative perceptual baselines:

- `speech_0021_6cd35984e877_seg_001__identity_roundtrip_ceiling.wav`
- `speech_0022_ba721f6129b9_seg_005__real_residual_resynthesis.wav`

Prior phase/magnitude swapping established on `speech_0021` that candidate STFT magnitude combined with
the real residual phase sounds correct. Therefore candidate magnitude and the fixed minimum-phase
renderer are not the dominant source of the remaining artifact.

## Temporal phase increment listening result

The owner listened to the candidate-magnitude controls produced with the target real-residual temporal
phase increments.

### Candidate initial per-bin anchor

`speech_0021_6cd35984e877_seg_001__candidate_mag_target_dphase_candidate_anchor_render.wav`

Owner listening result: sounds good, almost very good. Remaining defect is small but audible: a light
"trabada" voice quality plus slight wind/air texture.

### Zero initial per-bin anchor

`speech_0021_6cd35984e877_seg_001__candidate_mag_target_dphase_zero_anchor_render.wav`

Owner listening result: thinner, a little robotic, with telephone/cellphone-like voice and moving-air
texture.

## Objective context

For the rejected candidate residual versus the real residual:

- `speech_0021` temporal phase-increment alignment score: `0.4104`
- `speech_0021` temporal phase-increment circular MAE: `1.0396 rad`
- `speech_0022` temporal phase-increment alignment score: `0.3954`
- `speech_0022` temporal phase-increment circular MAE: `1.0579 rad`

These metrics are diagnostic only. Human listening remains the acceptance authority.

## Interpretation

This is a genuine perceptual advance but not a product PASS.

The experiment establishes that:

1. candidate magnitude can be retained;
2. the fixed minimum-phase renderer can be retained;
3. target-like temporal phase evolution removes most of the previously dominant robotic/radio defect;
4. the initial phase anchor across frequency is perceptually material, because the candidate anchor is
   clearly better than a zero anchor;
5. the remaining small wind/telephone/trabada texture is now localized to the cross-frequency structure
   of that phase anchor (equivalently its group-delay / frequency-phase field), not to magnitude or the
   renderer.

## Active next gate

No training is authorized yet. The next diagnostic varies only the first-frame phase anchor while
holding candidate magnitude and target temporal phase increments fixed:

`scripts/diagnostic_candidate_magnitude_phase_anchor_group_delay_v1.py`

It compares candidate anchor, full target anchor ceiling, smooth low/high band anchor swaps, and a
smooth low-dimensional phase-offset/group-delay correction. The purpose is to determine the minimum
phase-anchor representation required to remove the remaining small artifact before any new model is
designed or trained.
