# Boundary-safe speech alignment v2

## Root cause

The first duration cache (`alignment-v1`) converted every CTC blank frame into a neighboring
spoken token. That is acceptable for blank runs between phonemes, but it is wrong at recording
boundaries: leading silence was assigned to the first spoken token and trailing silence to the
last spoken token.

The forensic audit on the real LYKENOX corpus found:

- 132 duration records reviewed
- 25 long non-pause token outliers
- 23 affected utterances
- 20/25 outlier tokens at the first/last spoken-token boundary
- boundary fraction: 0.80
- diagnosis: `boundary_silence_absorption_likely`

This is an algorithmic duration-label issue, not evidence that the validated epoch-18 aligner
checkpoint must be retrained.

## v2 policy

`alignment-v2` keeps the same validated CTC/Viterbi path but changes only frame ownership:

```text
leading CTC blank run  -> <bos> duration
interior CTC blank run -> split between neighboring acoustic targets
trailing CTC blank run -> <eos> duration
```

Consequences:

1. The first and last spoken phonemes no longer inherit recording-boundary silence.
2. Every original mel frame is still accounted for exactly.
3. The acoustic model can learn boundary silence from structural BOS/EOS tokens.
4. `<wb>` remains context-only with zero acoustic duration.
5. Pause tokens created from punctuation remain real acoustic targets.
6. No change is made to the trained aligner weights.

## Artifact versioning

The old cache remains intact under `alignment-v1`. The corrected cache is generated under:

```text
datasets/lykenox/identity_voice/features/speech/alignment-v2/
```

Each v2 record stores explicit:

```json
"boundary_frames": {
  "leading": 0,
  "trailing": 0
}
```

and the complete `durations` vector maps those frames to BOS/EOS while `content` contains only
real acoustic content/pause targets.

## Regeneration gate

Use the already validated `best.pt`; do not retrain the aligner for this fix:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_boundary_fix_regenerate
```

The command:

1. loads the existing epoch-18 `best.pt`;
2. generates `alignment-v2` train/validation duration caches;
3. runs the outlier review on v2;
4. emits a compact final gate report.

If no >100-frame non-pause outliers remain, the next gate is `aligned_acoustic_smoke`.
If v2 still has boundary-heavy outliers, they are classified as residual alignment errors rather
than re-triggering the already-fixed blank-assignment bug.
