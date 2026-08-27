# Boundary-safe duration regeneration: timeout recovery

The `alignment-v2` boundary policy remains:

```text
leading CTC blank run  -> BOS duration
interior CTC blank run -> neighboring acoustic tokens
trailing CTC blank run -> EOS duration
```

The validated epoch-18 aligner checkpoint is not retrained for this operation.

## Why the first regeneration exceeded the executor limit

Duration regeneration performs aligner inference plus a monotonic Viterbi search for every
speech utterance. The original Viterbi implementation updated every CTC state inside nested
Python loops, which made full-corpus regeneration unnecessarily slow on the target CPU.
The command also only emitted its final report after all 132 utterances were complete.

## Current implementation

The Viterbi recurrence is now vectorized across CTC states while preserving the same legal
transitions and tie ordering. Backtracking remains sequential only over time steps.

Duration generation is also resumable:

- each completed utterance is written immediately as an `alignment-v2` `.pt` record;
- an existing record is reused only when cache version, frontend version, checkpoint SHA-256,
  utterance id, text, and token ids still match;
- `duration_progress.json` is written even when the run stops at its wall-clock budget;
- the boundary regeneration command uses an 85-second default budget so it can return before
  short external executor timeouts;
- rerunning the exact same command computes only missing utterances;
- a PID sentinel rejects duplicate concurrent regeneration processes and removes stale locks
  left by a killed process.

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_boundary_fix_regenerate
```

Possible clean outcomes:

- `status: pass` -> all train/validation records exist and the outlier review is clean;
- `status: review_required` -> all records exist, but residual real alignment outliers remain;
- `status: incomplete` -> the time budget was reached; rerun the same command to resume;
- `status: duration_generation_failed` -> at least one actual alignment error must be reviewed.

No acoustic training is allowed until regeneration is complete and the resulting outlier
report has been evaluated.
