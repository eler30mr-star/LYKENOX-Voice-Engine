# LYKENOX Speech v0

## Purpose

This milestone starts the first neural speech stack owned by LYKENOX Voice Engine.
It is intentionally independent of Piper, Coqui, OpenUtau, hosted APIs, reference-audio
cloning, and third-party TTS executables.

Product contract:

```text
Spanish text -> LYKENOX frontend -> LYKENOX acoustic model -> LYKENOX vocoder -> speech.wav
```

The current v0 implementation covers the frontend contract, the acoustic text-to-mel
core, real-data mel caching, and the first LYKENOX-owned alignment path. It does **not**
yet claim finished TTS quality because production alignment training, the vocoder, export,
and perceptual validation are still gates.

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

The initial configuration is a feasibility baseline, not a frozen production architecture.

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

`spanish_text_frontend.py` provides a stable `SpanishTextFrontend` contract with:

- NFC normalization
- lowercase normalization
- whitespace normalization
- deterministic Spanish grapheme vocabulary
- BOS/EOS/PAD/UNK/SPACE tokens

This remains an explicit bootstrap. A production Spanish phoneme/G2P implementation may
replace the internals later, but it must preserve the LYKENOX-owned frontend contract and
must not create a TTS-product runtime dependency.

## Real-data feature pipeline

The real speech dataset now passes through:

```text
LYKENOX WAV -> LYKENOX audio I/O -> 24 kHz -> 80-bin log-mel -> local feature cache
```

The train split currently contains:

- 118 utterances
- 119,897 mel frames
- cache path under `datasets/lykenox/identity_voice/features/speech/mel-v1/train`

Audio decoding is owned by the LYKENOX audio boundary and does not require TorchCodec.

## Alignment decision

Uniform token durations were used only for plumbing in the first real-data acoustic smoke
test. They are explicitly forbidden for production training.

The next alignment architecture is LYKENOX-owned and training-only:

```text
real mel
  -> compact convolutional acoustic frontend
  -> bidirectional GRU
  -> CTC token posteriors
  -> LYKENOX monotonic Viterbi forced alignment
  -> exact mel-frame durations per text token
```

Files:

- `lykenox_voice_engine/models/speech/alignment.py`
- `lykenox_voice_engine/core/ctc_alignment.py`
- `lykenox_voice_engine/training/speech_alignment_smoke.py`

Why this route:

- it trains directly from LYKENOX text + LYKENOX audio
- it does not need an external TTS or ASR executable
- it enforces monotonic text/audio order
- repeated symbols are handled through the CTC blank state
- every mel frame is assigned to a neighboring content token
- derived durations can later supervise the acoustic model's duration predictor
- the aligner can remain a training tool and does not have to ship in the final inference runtime

The first aligner uses a 2-frame acoustic stride for CPU efficiency. The forced-alignment
stage expands the result back to exact original mel-frame counts.

## Alignment smoke gate

Before training or caching production durations, run only the controlled real-data smoke:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_alignment_smoke --steps 120
```

The gate reports:

- aligner parameter count
- CTC training loss
- a fixed probe utterance CTC loss before/after training
- CPU step time
- monotonic forced-alignment score
- whether derived durations sum exactly to the probe mel-frame count
- whether every content token receives non-zero duration

Pass criteria:

1. all losses and gradients remain finite
2. fixed-probe CTC loss decreases
3. forced alignment returns a legal monotonic path
4. duration sum equals the original mel-frame count
5. all content tokens receive non-zero duration

This smoke model is not saved as a production aligner.

## Hard gates before real acoustic training

Do not start a long acoustic run until all of these are satisfied:

1. synthetic CPU probe passes
2. real mel cache is complete
3. real-data acoustic smoke test decreases loss without NaN/Inf
4. CTC alignment smoke gate passes
5. a persistent aligner checkpoint is trained and validated on held-out speech
6. production duration caches are generated and audited
7. the acoustic model is re-smoked using real aligned durations instead of uniform durations
8. a LYKENOX-owned vocoder path is designed and separately benchmarked
9. checkpoint/export manifest is versioned

## What comes next

Professional next milestones, in order:

1. Run the CTC alignment smoke on the actual laptop.
2. If it passes, train a small persistent LYKENOX aligner with train/validation metrics.
3. Generate deterministic duration caches for the speech corpus.
4. Audit alignment outliers before acoustic training.
5. Re-run the acoustic smoke test with real durations.
6. Design and benchmark a compact LYKENOX vocoder.
7. Train a short experimental speech model only after timing/memory/quality gates pass.
8. Export a self-contained LYKENOX artifact and connect it to `/speak`.

## Success definition

Speech v0 is not considered a usable TTS until unseen Spanish text can produce intelligible
speech with the persistent LYKENOX identity without reference audio, source conversion,
network access, or a third-party TTS executable.
