# LYKENOX Speech v0

## Purpose

This milestone builds the first neural speech stack owned by LYKENOX Voice Engine.
It is intentionally independent of Piper, Coqui, OpenUtau, hosted APIs, reference-audio
cloning, and third-party TTS executables.

Product contract:

```text
Spanish text -> LYKENOX frontend -> LYKENOX acoustic model -> LYKENOX vocoder -> speech.wav
```

The current v0 implementation covers the versioned Spanish phoneme frontend, real-data
mel caching, an owned persistent CTC/Viterbi aligner, cleaned `alignment-v3` duration
supervision, and the first aligned acoustic optimization gate. It does **not** yet claim
finished TTS quality because the batched training contract, vocoder, long identity
training, export, and perceptual validation are still gates.

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
- tensorized duration-conditioned length regulator
- mel decoder
- default 80-bin mel output
- 24 kHz target sample rate

The length regulator no longer uses Python loops or duration-dependent `.item()` calls.
It emits exact per-item mel lengths and a frame mask for padded batches. Teacher durations
are never clipped by the inference-only `max_duration_frames` safety bound.

Measured on the target CPU during the synthetic gate:

- parameters: 1,967,889 with the original temporary vocabulary size
- 10 forward/backward/update steps: pass
- mean step time: 0.0352 seconds

Measured during the first real-data plumbing smoke with temporary uniform durations:

- 118 training utterances available
- 20 update steps: pass
- first loss: 1.332389
- last loss: 1.029874
- mean step time: 0.0668 seconds

Uniform durations were plumbing-only and are forbidden for production training.

## Spanish frontend

The product frontend is versioned as:

```text
es-phoneme-v1
```

`spanish_text_frontend.py` exposes the stable product interface and `spanish_g2p.py`
contains LYKENOX-owned pronunciation rules.

Current properties:

- NFC/lowercase/whitespace normalization
- deterministic Latin-American Spanish seseo/yeismo baseline
- phoneme tokens instead of temporary raw-grapheme training tokens
- local handling of `ch`, `ll`, `rr`, `ñ`, `qu`, `gue/gui`, `güe/güi`, soft `c/g`, `j`,
  `b/v`, `z`, and silent `h`
- explicit `<wb>` word-boundary context/timing token
- explicit `<pau_short>` and `<pau_long>` prosodic tokens from punctuation
- deterministic bootstrap number handling
- no external phonemizer or TTS frontend dependency

The exact vocabulary is now bound into acoustic checkpoints together with a SHA-256
checksum. Acoustic model construction uses `SpanishTextFrontend().vocab_size`; the old
hard-coded 128-symbol assumption is not part of the long-run training contract.

## Real-data feature pipeline

```text
LYKENOX WAV -> LYKENOX audio I/O -> 24 kHz -> 80-bin log-mel -> local feature cache
```

The train split contains:

- 118 utterances
- 119,897 mel frames
- cache under `datasets/lykenox/identity_voice/features/speech/mel-v1/train`

Audio decoding is owned by the LYKENOX audio boundary and does not require TorchCodec.
The mel cache is audio-only; token IDs are generated dynamically from the versioned
frontend.

## Persistent owned alignment

Training-time alignment is:

```text
real mel
  -> compact LYKENOX acoustic aligner
  -> CTC phoneme posteriors
  -> LYKENOX monotonic Viterbi path
  -> LYKENOX timing ownership policy
  -> exact model-token durations
```

The `es-phoneme-v1` alignment smoke passed on the target laptop:

- parameters: 253,855
- 120 CPU steps
- first training loss: 15.49226
- last training loss: 2.79187
- fixed-probe CTC loss: 15.493403 -> 2.729163
- mean step time: 0.2904 seconds
- exact duration coverage: pass
- every aligned content token non-zero: pass

The persistent aligner checkpoint recovered from the interrupted long run passed held-out
validation:

- best checkpoint epoch: 18
- stored/recomputed validation CTC loss: 1.033748
- random-initialization validation CTC loss: 13.918439
- held-out forced-alignment success: 12/12
- held-out exact duration coverage: 12/12
- held-out content tokens non-zero: 12/12

## alignment-v3 timing contract

Two mechanical duration bugs were found and corrected without retraining the validated
aligner:

1. leading/trailing blank runs were being folded into the first/last phoneme;
2. long interior blank runs between words were being split into neighboring phonemes.

The current timing policy is:

```text
leading blank              -> <bos>
trailing blank             -> <eos>
blank between words        -> <wb>
blank adjacent to pause    -> <pau_short>/<pau_long>
intra-word blank           -> split between neighboring phonemes
```

The final cleaned cache is versioned as `alignment-v3` and was generated for all 132
utterances:

- train: 118/118
- validation: 14/14
- outlier tokens above 100 frames: 0
- outlier utterances: 0
- non-pause duration median: 5 frames
- non-pause duration p95: 10 frames
- non-pause duration max: 99 frames

The validated aligner remains a training tool; normal product inference does not require it.

## Aligned acoustic gate

The first acoustic smoke using real cleaned durations passed on CPU:

- alignment: `alignment-v3`
- items used: 8
- steps: 40
- parameters: 1,949,073 with the exact `es-phoneme-v1` vocabulary size
- first/last total training loss: 1.411191 -> 0.604989
- first/last acoustic loss: 1.309268 -> 0.583238
- first/last duration loss: 1.019234 -> 0.217505
- fixed-probe total loss: 1.411519 -> 0.766106
- fixed-probe acoustic loss: 1.310069 -> 0.744595
- fixed-probe duration loss: 1.014506 -> 0.215107
- mean CPU step time: 0.1265 seconds

This proves the real path:

```text
real WAV -> mel-v1 -> alignment-v3 -> es-phoneme-v1 -> acoustic model -> loss -> backward -> update
```

It does not prove final identity, intelligibility, waveform quality, or long-run training
stability.

## Batched acoustic training contract

Long training is blocked until the explicit batching/checkpoint gate passes. The contract
now lives in:

- `training/speech_aligned_data.py`: alignment-v3 dataset, padded batches, token/mel masks
- `training/speech_losses.py`: masked acoustic and duration losses
- `models/speech/network.py`: tensorized length regulation and exact frame masks
- `training/speech_acoustic_artifact.py`: versioned checkpoint + exact vocabulary/provenance
- `training/speech_training_contract_smoke.py`: short CPU gate for all of the above

A training checkpoint records:

- checkpoint format version and kind
- `es-phoneme-v1` frontend version
- complete vocabulary and vocabulary SHA-256
- exact acoustic model configuration
- epoch/global step/validation metric
- train and validation manifest SHA-256 hashes
- alignment-v3 duration-audit SHA-256
- mel-cache version
- model state and optional optimizer state

Padded mel frames must contribute exactly zero acoustic loss, and padded text tokens must
contribute exactly zero duration loss.

## Hard gates before a long acoustic run

1. synthetic CPU probe: **passed**
2. real mel cache: **passed**
3. real-data plumbing smoke: **passed**
4. `es-phoneme-v1`: **active**
5. CTC alignment smoke: **passed**
6. persistent aligner held-out validation: **passed**
7. complete duration cache: **passed**
8. duration outlier correction/audit: **passed with alignment-v3**
9. aligned acoustic smoke: **passed**
10. batched/masked/export-oriented training-contract smoke: **next gate**
11. LYKENOX-owned vocoder CPU feasibility benchmark: pending
12. bounded short identity-training experiment: pending
13. runtime export and `/speak` integration: pending

Do not start a long acoustic run before gates 10 and 11 pass.

## Next controlled command

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_training_contract_smoke
```

If it passes, the next architectural gate is a bounded CPU benchmark of a compact
LYKENOX-owned vocoder path. A long identity run is still not authorized at that point.

## Success definition

Speech v0 is not considered usable TTS until unseen Spanish text produces intelligible
speech with the persistent LYKENOX identity without reference audio, source conversion,
network access, or a third-party TTS executable.