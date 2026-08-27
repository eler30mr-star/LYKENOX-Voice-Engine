# LYKENOX Speech Alignment Pipeline

## Decision

The production speech training path uses the LYKENOX-owned Spanish phoneme frontend plus a compact LYKENOX CTC aligner and local monotonic Viterbi decoding. The aligner is a training-time component only; it is not a dependency of the final `/speak` runtime.

```text
Spanish text
  -> es-phoneme-v1
  -> phoneme IDs

LYKENOX WAV
  -> log-mel
  -> LYKENOX CTC aligner
  -> CTC posteriors
  -> LYKENOX monotonic Viterbi
  -> exact mel-frame durations
```

No Piper, Coqui, Whisper, external TTS executable, hosted API, source speaker, reference WAV, RVC, or SVC is used by this path.

## Gates already passed on the target CPU

Acoustic model synthetic probe:

- status: pass
- parameters: 1,967,889
- mean step time: 0.0352 s

Real-data acoustic plumbing smoke:

- status: pass
- 118 training utterances available
- first loss: 1.332389
- last loss: 1.029874
- mean step time: 0.0668 s

Phoneme CTC/Viterbi alignment smoke after switching to `es-phoneme-v1`:

- status: pass
- parameters: 253,855
- steps: 120
- first training loss: 15.49226
- last training loss: 2.79187
- fixed probe CTC loss before: 15.493403
- fixed probe CTC loss after: 2.729163
- mean step time: 0.2904 s
- exact duration sum: true
- every aligned content token non-zero: true
- observed content duration range in the probe: 2 to 250 mel frames

The smoke checkpoint is deliberately discarded. These numbers validate the mechanism, not production alignment quality.

## Persistent aligner artifact

`alignment_artifact.py` defines a versioned checkpoint contract. A checkpoint records:

- artifact version and kind
- `es-phoneme-v1` frontend version
- exact LYKENOX vocabulary
- aligner configuration
- acoustic feature configuration
- epoch and validation CTC loss
- model state
- reproducibility metadata

Loading rejects a checkpoint whose frontend version or vocabulary does not match the current LYKENOX frontend. This prevents silently training the acoustic model with stale duration caches.

Default local checkpoint location:

```text
models/lykenox_identity/training/speech_aligner/es-phoneme-v1/
  best.pt
  last.pt
  training_report.json
  alignment_pipeline_report.json
```

These generated model artifacts remain local and are not source-controlled.

## Persistent training gate

Run the controlled pipeline:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_alignment_pipeline --epochs 20 --patience 4
```

The training stage:

1. loads the LYKENOX train and validation manifests;
2. reuses the existing mel cache and creates missing validation mels locally;
3. filters only items that cannot form a legal CTC path or exceed the explicit smoke/training length ceiling;
4. trains with deterministic shuffling and gradient clipping;
5. evaluates the full eligible validation split after every epoch;
6. saves `best.pt` only when validation CTC improves;
7. applies early stopping;
8. reloads the best checkpoint and audits held-out forced alignments.

The training gate passes only if validation CTC improves and every eligible held-out item can produce a legal forced alignment whose durations cover the complete mel sequence with non-zero content durations.

## Duration cache and audit

Only after the persistent aligner gate passes, the same command generates deterministic duration caches for both train and validation data.

The cache is keyed by:

- cache format `alignment-v1`;
- frontend version `es-phoneme-v1`;
- SHA-256 of the exact best aligner checkpoint.

Target layout:

```text
datasets/lykenox/identity_voice/features/speech/alignment-v1/
  es-phoneme-v1/
    <checkpoint-sha-prefix>/
      train/
      val/
      duration_audit.json
```

Each utterance record contains the original token IDs, exact duration vector, mel frame count, token-level audit rows, frontend version, and full checkpoint hash. This makes stale or mismatched durations detectable instead of silently reusable.

The audit reports:

- generated/failed items per split;
- mean alignment score per split;
- content-duration median, p95, and maximum;
- non-pause duration median, p95, and maximum;
- suspicious utterances with unusually long non-pause tokens;
- exact frame-coverage failures.

Long non-pause durations are warnings rather than automatic rejection because the threshold is a diagnostic gate, not a linguistic truth. The distribution must be reviewed before the acoustic model is trained on these durations.

## Next gate

Do not start long acoustic-model training immediately after the pipeline. First inspect the duration audit. The next implementation stage is to bind the acoustic dataset to the checkpoint-versioned duration cache, replace the old uniform-duration smoke path, and run a short aligned acoustic smoke test.
