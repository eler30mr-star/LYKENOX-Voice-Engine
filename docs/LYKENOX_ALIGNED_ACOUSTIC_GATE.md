# LYKENOX aligned acoustic smoke gate

The cleaned speech timing contract is now `alignment-v3`.

Validated local dataset state before this gate:

- persistent CTC aligner: recovered `best.pt`, epoch 18
- train durations: 118/118
- validation durations: 14/14
- duration cache: `alignment-v3`
- boundary policy: leading blank -> BOS, trailing blank -> EOS
- interior policy: word-boundary blank -> `<wb>`, pause-adjacent blank -> pause token,
  intra-word blank -> neighboring phonemes
- non-pause outliers above 100 frames: 0
- non-pause duration median: 5 frames
- non-pause p95: 10 frames
- non-pause max: 99 frames

The next gate is intentionally a short CPU optimization smoke, not a long training run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_aligned_smoke
```

Unlike the old real-data plumbing smoke, this command has **no uniform-duration fallback**.
It loads `alignment-v3`, verifies the cached text/token identity, requires the teacher
Durations to sum exactly to the mel-frame count, and computes acoustic loss over the full
mel target without clipping.

The acoustic model contract was corrected before enabling this gate: externally supplied
teacher durations are no longer clamped by `max_duration_frames`. That limit belongs only
to predicted inference durations. Clipping aligned teacher durations would silently change
the regulated mel length and corrupt supervision.

The smoke reports a fixed probe before/after training for total, acoustic, and log-duration
losses. A pass requires all three probe losses to decrease with finite gradients and exact
mel-length preservation.

Passing this gate proves only that the current compact LYKENOX acoustic prototype can learn
from the cleaned real alignment contract on the target CPU. It does not yet authorize a
long identity-training run. Before that run, LYKENOX still needs a production batching and
mel-mask contract, an export-safe length regulator, exact vocabulary/checkpoint metadata,
and a separately benchmarked owned vocoder path.
