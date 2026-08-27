# LYKENOX Aligner Interruption Recovery

## Why this exists

Persistent aligner training can outlive an external terminal/executor time limit on a
CPU-only machine. A timeout is not itself a model-quality failure if the trainer already
wrote `best.pt` and `last.pt`.

LYKENOX must recover deterministically instead of discarding useful local work or blindly
starting another long run.

## Recovery rule

When `best.pt` exists, the alignment pipeline audits it before any retraining.

The recovery gate:

1. loads the versioned LYKENOX aligner artifact;
2. rebuilds the held-out validation dataset with the current `es-phoneme-v1` frontend;
3. reconstructs the deterministic random-initialization validation baseline;
4. recomputes validation CTC loss for `best.pt`;
5. runs forced CTC/Viterbi alignment across every eligible validation utterance;
6. requires exact mel-frame duration coverage and non-zero acoustic-token durations;
7. writes `recovery_validation_report.json`;
8. only after that gate passes, generates train/validation duration caches.

No training occurs during recovery.

## Command

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_aligner_recover
```

Outputs are written beside the aligner checkpoints under:

```text
models/lykenox_identity/training/speech_aligner/es-phoneme-v1/
```

Expected reports:

- `recovery_validation_report.json`
- `recovery_pipeline_report.json`

## Status semantics

- `pass`: recovered checkpoint passes validation and duration cache has no configured
  outlier warnings; proceed to aligned acoustic smoke.
- `duration_review_required`: checkpoint is valid and duration generation completed, but
  one or more non-pause duration outliers require inspection before acoustic training.
- `duration_gate_failed`: at least one utterance could not produce a valid duration cache.
- `checkpoint_gate_failed`: the recovered checkpoint did not pass held-out validation;
  further aligner training is required.

## Product boundary

The aligner remains a training-only LYKENOX component. Recovery does not add an inference
runtime dependency and does not change the final product contract:

```text
text -> persistent LYKENOX speech model -> speech.wav
```
