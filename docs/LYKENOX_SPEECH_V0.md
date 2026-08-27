# LYKENOX Speech v0

## Purpose

This milestone starts the first neural speech stack owned by LYKENOX Voice Engine.
It is intentionally independent of Piper, Coqui, OpenUtau, hosted APIs, reference-audio
cloning, and third-party TTS executables.

Product contract:

```text
Spanish text -> LYKENOX frontend -> LYKENOX acoustic model -> LYKENOX vocoder -> speech.wav
```

The current v0 implementation covers the versioned Spanish phoneme frontend, the acoustic
text-to-mel core, real-data mel caching, and the first LYKENOX-owned alignment path. It
does **not** yet claim finished TTS quality because duration outlier correction, the
vocoder, export, and perceptual validation are still gates.

## Design rules

1. The master dataset remains engine-neutral.
2. LYKENOX owns the text/frontend contract, model configuration, model artifact layout,
   runtime interface, training orchestration, and API.
3. No reference WAV is required at normal inference.
4. The final installed product must synthesize offline from packaged artifacts.
5. Research architectures may inform implementation, but no third-party TTS executable is
   a required runtime component.
6. No source speaker, source singer, RVC, SVC, or post-conversion is part of the speech path.

## v0 acoustic architecture

`LykenoxSpeechAcousticModel` is deliberately compact and CPU-oriented:

- learned token embedding
- Transformer text encoder
- duration predictor
- deterministic length regulator
- mel decoder
- default 80-bin mel output
- 24 kHz target sample rate

Measured on the target CPU during the synthetic gate:

- parameters: 1,967,889
- 10 forward/backward/update steps: pass
- mean step time: 0.0352 seconds

Measured during the first real-data acoustic smoke test:

- 118 training utterances available
- 20 update steps: pass
- first loss: 1.332389
- last loss: 1.029874
- mean step time: 0.0668 seconds

These results prove mechanical local CPU viability for the current acoustic core. They do
not prove final voice quality.

## Spanish frontend

The product frontend is now versioned as:

```text
es-phoneme-v1
```

`spanish_text_frontend.py` exposes the stable product interface and
`spanish_g2p.py` contains LYKENOX-owned pronunciation rules.

Current properties:

- NFC/lowercase/whitespace normalization
- deterministic Latin-American Spanish seseo/yeismo baseline
- phoneme tokens instead of temporary raw-grapheme training tokens
- local handling of `ch`, `ll`, `rr`, `ñ`, `qu`, `gue/gui`, `güe/güi`, soft `c/g`, `j`,
  `b/v`, `z`, and silent `h`
- explicit `<wb>` word-boundary context token
- explicit `<pau_short>` and `<pau_long>` prosodic tokens from punctuation
- deterministic bootstrap number handling
- no external phonemizer or TTS frontend dependency

The rule set is versioned so pronunciation improvements can be introduced later without
silently changing an already-trained model artifact.

## Real-data feature pipeline

The real speech dataset passes through:

```text
LYKENOX WAV -> LYKENOX audio I/O -> 24 kHz -> 80-bin log-mel -> local feature cache
```

The train split currently contains:

- 118 utterances
- 119,897 mel frames
- cache path under `datasets/lykenox/identity_voice/features/speech/mel-v1/train`

Audio decoding is owned by the LYKENOX audio boundary and does not require TorchCodec.
The mel cache remains reusable after the phoneme frontend change because it contains audio
features only; token IDs are generated dynamically from the text.

## Alignment decision

Uniform token durations were used only for plumbing in the first real-data acoustic smoke
test. They are explicitly forbidden for production training.

The LYKENOX-owned training-time alignment architecture is:

```text
real mel
  -> compact convolutional acoustic frontend
  -> bidirectional GRU
  -> CTC phoneme posteriors
  -> LYKENOX monotonic Viterbi forced alignment
  -> exact mel-frame durations per acoustic token
```

Files:

- `lykenox_voice_engine/models/speech/alignment.py`
- `lykenox_voice_engine/core/ctc_alignment.py`
- `lykenox_voice_engine/training/speech_alignment_smoke.py`

The `es-phoneme-v1` alignment smoke passed on the target laptop:

- parameters: 253,855
- 120 CPU steps
- first training loss: 15.49226
- last training loss: 2.79187
- fixed-probe CTC loss: 15.493403 -> 2.729163
- mean step time: 0.2904 seconds
- exact duration coverage: pass
- every aligned content token non-zero: pass

`<wb>` is encoder context only and is excluded from CTC targets. Pause tokens remain
alignable because they represent real acoustic/prosodic time.

## Persistent aligner recovery

The first persistent run was interrupted by the local executor timeout after it had already
written checkpoints. The existing `best.pt` was recovered instead of discarding work.
The recovery gate validated:

- best checkpoint epoch: 18
- stored validation CTC loss: 1.033748
- recomputed validation CTC loss: 1.033748
- random-initialization validation CTC loss: 13.918439
- held-out forced-alignment success: 12/12
- held-out exact duration coverage: 12/12
- held-out content tokens non-zero: 12/12

The recovered checkpoint generated duration caches successfully for:

- train: 118/118, 0 failures
- validation: 14/14, 0 failures

Those facts validate the checkpoint and cache generation mechanically, but the duration
audit found 23 utterances containing at least one non-pause token above the current
100-frame warning threshold. Acoustic training remains blocked until those outliers are
explained or corrected.

## Duration outlier gate

The current CTC blank-to-token policy has a specific boundary risk: a leading CTC blank
run can be assigned to the first acoustic token and a trailing blank run can be assigned
to the last acoustic token. If recordings contain boundary silence, this can create an
artificially long first or last phoneme even when the monotonic path itself is legal.

Before changing the aligner or deleting data, run the fast cache-only forensic review:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_duration_outlier_review
```

It performs no training and no neural inference. It classifies every >100-frame non-pause
outlier as first, last, or interior, reports milliseconds and token-relative statistics,
and writes `duration_outlier_review.json` inside the active duration cache. A boundary-heavy
result supports correcting blank-boundary handling and regenerating durations from the
same validated epoch-18 checkpoint; an interior-heavy result requires inspecting actual
alignment failures instead.

## Hard gates before real acoustic training

Do not start a long acoustic run until all of these are satisfied:

1. synthetic CPU probe passes
2. real mel cache is complete
3. real-data acoustic smoke test decreases loss without NaN/Inf
4. `es-phoneme-v1` pronunciation tests pass
5. CTC alignment smoke passes using `es-phoneme-v1`
6. persistent aligner checkpoint passes held-out validation
7. production duration caches are generated without alignment failures
8. duration outliers are explained/corrected and re-audited
9. the acoustic model is re-smoked using real aligned durations instead of uniform durations
10. a LYKENOX-owned vocoder path is designed and separately benchmarked
11. checkpoint/export manifest is versioned

## What comes next

Professional next milestones, in order:

1. Run the cache-only duration outlier forensic review.
2. Correct boundary blank assignment if the outliers are boundary-heavy; otherwise inspect interior failures.
3. Regenerate/audit duration caches from the already validated epoch-18 checkpoint.
4. Re-run the acoustic smoke test with real durations.
5. Design and benchmark a compact LYKENOX vocoder.
6. Train a short experimental speech model only after timing/memory/quality gates pass.
7. Export a self-contained LYKENOX artifact and connect it to `/speak`.

## Success definition

Speech v0 is not considered a usable TTS until unseen Spanish text can produce intelligible
speech with the persistent LYKENOX identity without reference audio, source conversion,
network access, or a third-party TTS executable.
