# LYKENOX alignment-v3 timing policy

The validated epoch-18 CTC aligner is not retrained for this change. `alignment-v3`
changes only how a legal monotonic CTC path is converted into durations for the acoustic
model.

## Why v3 exists

`alignment-v2` fixed recording-boundary contamination by preserving leading CTC blank time
on `<bos>` and trailing blank time on `<eos>`. The subsequent forensic audit found that the
remaining interior duration outliers were dominated by CTC blank allocation rather than
direct phoneme occupancy.

A long CTC blank between words is timing/silence information. Splitting it into the two
neighboring phonemes teaches the acoustic model an incorrect phoneme duration.

## Policy

```text
leading blank                    -> <bos>
trailing blank                   -> <eos>
interior blank crossing <wb>     -> <wb>
interior blank adjacent to pause -> <pau_short>/<pau_long>
intra-word blank                 -> split between neighboring phonemes
```

`<wb>` remains excluded from the CTC target sequence. It is now allowed to carry duration
in the acoustic timing sequence, which gives word-level silence/prosodic gaps an explicit
LYKENOX-owned timing carrier without changing the aligner vocabulary or invalidating the
validated checkpoint.

The intra-word fallback is deliberately conservative: no hidden silence token is invented
inside a word. Long intra-word cases remain visible to the outlier gate for further review.

## Versioning

The new records are stored under `alignment-v3`. `alignment-v1` and `alignment-v2` are not
overwritten. Every record remains bound to the exact aligner checkpoint SHA, frontend
version, text, token IDs, and timing-policy version.

## Gate

Regenerate and immediately re-audit with:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_interior_fix_regenerate
```

No aligner training is performed by this command. Acoustic training remains blocked until
the v3 duration audit is accepted.
