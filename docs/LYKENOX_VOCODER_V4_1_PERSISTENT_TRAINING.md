# LYKENOX Vocoder v4.1 — bounded resumable persistent training

## Decision entering this gate

`lykenox_pitch_source_filter_v4_1` is the first LYKENOX vocoder candidate that has cleared
all of the architecture-selection gates accumulated during v0-v4.1:

- no generated-specific `24 kHz / 256 = 93.75 Hz` frame-grid carrier
- no sub-bass/silence collapse
- useful energy above 300 Hz
- generated held-out F0 close to paired real references
- improved target-relative spectral balance
- human listening showed recognizable voice-related harmonic/acoustic structure rather than
  only a synthetic carrier
- the v4.1 checkpoint smoke passed exact state/waveform round-trip, provenance, architecture,
  both optimizer states and resume metadata

This does **not** make v4.1 the final runtime artifact. It makes v4.1 the architecture family
worth training persistently. The next engineering problem is reliable training, not another
vocoder redesign.

## Product boundary remains unchanged

Training-time isolation uses F0 and voicing extracted from owned target WAVs. Final product
inference will not require a WAV reference.

```text
text
  -> LYKENOX Spanish frontend
  -> LYKENOX acoustic model
       mel + duration/prosody + predicted F0 + predicted voicing
  -> LYKENOX v4.1 source-filter vocoder
  -> waveform
```

No external TTS/SVS backend, source speaker, voice-conversion stage, hosted API, account or
normal-inference model download is introduced by this trainer.

## Trainer contract

Module:

```text
lykenox_voice_engine.training.speech_vocoder_source_filter_train
```

Contract version:

```text
source-filter-bounded-resumable-v1
```

Default persistent run:

- CPU only
- `64` mel frames per segment (`~0.683 s`)
- up to `118` train utterances and `14` held-out validation utterances
- `24` maximum epochs
- first `8` epochs: reconstruction + target-relative source-balance objective
- remaining epochs: same objectives plus mild adversarial/feature-matching pressure
- early stopping patience `6`
- best model chosen by held-out
  `reconstruction + 0.50 * spectral_balance`
- command wall-clock budget `70 s`
- `8 s` checkpoint reserve before that budget
- `last.pt` checkpoint at least every `16` updates
- rerunning the same command resumes the exact compatible run

The wall-clock budget is intentionally below the known external command cutoff. It is an
execution control, not part of experiment identity, so later invocations may use a
different wall-clock budget without invalidating the checkpoint.

## Dynamic but reproducible train coverage

The old short vocoder experiments repeatedly saw one fixed crop from each utterance. That is
not enough for persistent training.

The new trainer uses:

```text
epoch train crop seed = base seed + epoch number
```

and a deterministic shuffled order. Therefore each epoch sees a different reproducible
window from each available utterance. A mid-epoch resume regenerates the same crops and the
same order exactly.

Validation is different: its segment set is fixed for the entire run and its segment-set
SHA256 is stored in the run configuration. This keeps epoch-to-epoch validation comparable.

## Exact mid-epoch resume state

`last.pt` stores:

- v4.1 generator state
- discriminator state
- both AdamW optimizer states
- torch RNG state
- 1-based current/next epoch
- exact next shuffled item offset
- global update count
- full completed-epoch history
- partial-epoch reconstruction/balance/adversarial accumulator
- best score/epoch and early-stopping state
- exact training configuration
- dataset/frontend/pitch/source-balance provenance

Checkpoint writes performed by the trainer use a sibling temporary file and `os.replace` so
an interrupted write cannot replace the last known-good `last.pt` with a half-written file.

## Gate before real persistent training

First run the exact-resume smoke:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_source_filter_resume_smoke
```

It creates two isolated runs:

1. ten uninterrupted updates
2. three updates, checkpoint, process-level resume, seven more updates

It then requires exact equality of:

- generator state
- discriminator state
- generator optimizer
- discriminator optimizer
- torch RNG
- epoch/item/global-step position
- partial-epoch metric state
- completed history
- generated probe waveform

Expected next gate:

```text
run_bounded_resumable_v4_1_training
```

## Persistent command after the resume gate passes

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_source_filter_train
```

If the command returns:

```text
status: incomplete
next_gate: rerun_same_command_to_resume
```

run **the exact same command again**. It is not a failure; it is the normal bounded execution
contract.

Artifacts:

```text
models/lykenox_identity/training/vocoder_source_filter_v4_1/
  last.pt
  best.pt
  training_progress.json
  training_report.json
  listening/
```

Do not delete `last.pt` between bounded invocations.

## Completion gate

When training stops through max epochs or early stopping, the trainer reloads `best.pt`,
writes three held-out generated/reference pairs, and re-runs the known structural gates.

A persistent numerical pass requires:

```text
validation_selection_improved: true
validation_spectral_balance_improved: true
confirmed_generated_specific_frame_locks: 0
subbass_or_silence_collapse_count: 0
upper_voice_band_missing_count: 0
automatic_artifact_gate_pass: true
```

Even then, human listening remains mandatory. Only after the persistent listening gate is
accepted should v4.1 be prepared for runtime export and should the speech acoustic model be
extended with LYKENOX-owned F0/voicing prediction heads.
