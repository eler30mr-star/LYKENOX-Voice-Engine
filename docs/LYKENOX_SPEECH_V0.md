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
does **not** yet claim finished TTS quality because persistent alignment training, the
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

The first grapheme smoke passed on the target laptop:

- parameters: 257,725
- 120 CPU steps
- first training loss: 15.663699
- last training loss: 2.876315
- fixed-probe CTC loss: 15.665837 -> 2.9726
- mean step time: 0.7299 seconds
- exact duration coverage: pass
- every aligned content token non-zero: pass

That smoke proved the alignment mechanism, not a production checkpoint. Before persistent
alignment training, the frontend was deliberately upgraded to `es-phoneme-v1` so a long
run is not wasted on the temporary grapheme representation.

`<wb>` is encoder context only and is excluded from CTC targets. Pause tokens remain
alignable because they represent real acoustic/prosodic time.

## Hard gates before real acoustic training

Do not start a long acoustic run until all of these are satisfied:

1. synthetic CPU probe passes
2. real mel cache is complete
3. real-data acoustic smoke test decreases loss without NaN/Inf
4. `es-phoneme-v1` pronunciation tests pass
5. CTC alignment smoke passes again using `es-phoneme-v1`
6. a persistent aligner checkpoint is trained and validated on held-out speech
7. production duration caches are generated and audited
8. the acoustic model is re-smoked using real aligned durations instead of uniform durations
9. a LYKENOX-owned vocoder path is designed and separately benchmarked
10. checkpoint/export manifest is versioned

## What comes next

Professional next milestones, in order:

1. Re-run the controlled CTC/Viterbi smoke using `es-phoneme-v1`.
2. Train a small persistent LYKENOX aligner with train/validation metrics and checkpointing.
3. Generate deterministic duration caches for the complete speech corpus.
4. Audit alignment outliers before acoustic training.
5. Re-run the acoustic smoke test with real durations.
6. Design and benchmark a compact LYKENOX vocoder.
7. Train a short experimental speech model only after timing/memory/quality gates pass.
8. Export a self-contained LYKENOX artifact and connect it to `/speak`.

## Success definition

Speech v0 is not considered a usable TTS until unseen Spanish text can produce intelligible
speech with the persistent LYKENOX identity without reference audio, source conversion,
network access, or a third-party TTS executable.
