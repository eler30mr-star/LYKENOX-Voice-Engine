# LYKENOX Speech v0

## Purpose

This milestone starts the first neural speech stack owned by LYKENOX Voice Engine.
It is intentionally independent of Piper, Coqui, OpenUtau, hosted APIs, reference-audio
cloning, and third-party TTS executables.

Product contract:

```text
Spanish text -> LYKENOX frontend -> LYKENOX acoustic model -> LYKENOX vocoder -> speech.wav
```

The current v0 implementation covers the frontend contract and the acoustic text-to-mel
core. It does **not** yet claim finished TTS quality because the production aligner,
vocoder, real dataset trainer, export pipeline, and perceptual validation are still gates.

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
Parameter count and CPU step time must be measured on the target laptop before real
training is approved.

## Spanish frontend

`spanish_text_frontend.py` currently provides:

- NFC normalization
- lowercase normalization
- whitespace normalization
- deterministic Spanish grapheme vocabulary
- BOS/EOS/PAD/UNK/SPACE tokens

This is an explicit bootstrap. A production Spanish phoneme/G2P frontend may replace the
grapheme tokenizer later, but it must preserve the LYKENOX-owned frontend contract and
must not create a TTS-product runtime dependency.

## CPU feasibility gate

Before touching the real 23+ minute identity corpus, run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_cpu_probe
```

The probe performs forward + backward + optimizer steps on CPU and reports:

- parameter count
- mean/min/max seconds per step
- Torch/Python/platform
- synthetic loss
- best-effort memory measurement

This probe does **not** validate voice quality. It only answers whether the proposed
acoustic core can train mechanically on the target CPU.

## Hard gates before real training

Do not start a long run until all of these are satisfied:

1. CPU probe completes without OOM or unsupported operations.
2. Step time is measured and extrapolated.
3. Real speech dataset loader produces mel targets correctly.
4. Train/validation split remains stable.
5. A duration/alignment strategy is selected and tested.
6. A LYKENOX-owned vocoder path is designed and separately benchmarked.
7. Checkpoint format and export manifest are versioned.
8. A 10-50 step real-data smoke test decreases loss without NaN/Inf.

## What comes next

Professional next milestones, in order:

1. Run the synthetic CPU probe on the actual laptop.
2. Build the engine-neutral speech dataset loader and mel feature cache.
3. Implement/validate alignment or duration supervision.
4. Run a tiny real-data acoustic training smoke test.
5. Design and benchmark a compact LYKENOX vocoder.
6. Train a short experimental model only after timing/memory gates pass.
7. Export a self-contained LYKENOX artifact and connect it to `/speak`.

## Success definition

Speech v0 is not considered a usable TTS until unseen Spanish text can produce intelligible
speech with the persistent LYKENOX identity without reference audio, source conversion,
network access, or a third-party TTS executable.
