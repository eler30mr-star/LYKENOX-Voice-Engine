# LYKENOX Phase-Exclusive Handoff V3 — Listening Gate

Policy: LYX-POL-001
Date: 2026-09-03

## Execution status

The phase-exclusive handoff V3 held-out renderer completed successfully using the existing trained pitch-synchronous checkpoint. No training or optimizer step was executed and no checkpoint was written.

Reported execution contract:

- status: `ready_for_phase_exclusive_handoff_source_v3_listening`
- training_executed: `false`
- optimizer_created: `false`
- checkpoint_written: `false`
- raw_v2_pitch_sync_samplewise_mix_used: `false`
- source_authority: `pitch_sync_inside_complete_cycles_v2_elsewhere`
- handoff: `period_derived_cubic_hermite_c1_bridge`
- posthoc_gain_normalization_used: `false`
- posthoc_eq_used: `false`
- posthoc_denoising_used: `false`

## Held-out listening gate

The run is not accepted by metrics. Product/source quality remains undecided until complete human listening.

Primary speech_0021 comparison:

1. `speech_0021_6cd35984e877_seg_001__phase_exclusive_handoff_source_v3.wav`
2. `speech_0021_6cd35984e877_seg_001__v2_baseline_source.wav`
3. `speech_0021_6cd35984e877_seg_001__identity_roundtrip_ceiling.wav`
4. `speech_0021_6cd35984e877_seg_001__reference.wav`

Decision criterion: determine whether the terminal word/phrase chirp is materially reduced without losing the near-reference voice quality previously achieved by the pitch-synchronous source.

No additional architecture change, training, gain, EQ, denoise, source mixing, or tuning is authorized before this listening result is reported.
