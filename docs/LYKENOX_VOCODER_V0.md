# LYKENOX Vocoder v0

## Product boundary

The final speech product requires an owned local waveform stage:

```text
LYKENOX acoustic mel -> LYKENOX vocoder -> 24 kHz waveform
```

The installed product must not require an external vocoder executable, hosted API,
reference speaker, source voice, or model download during normal inference. The runtime
ships the LYKENOX generator only; the discriminator described below is training-only.

## First CPU feasibility architecture

`LykenoxVocoderGenerator` is a compact non-autoregressive PyTorch generator implemented
inside this repository. It uses a mel pre-convolution, three learned upsampling stages,
local residual convolution blocks, and a bounded waveform output. The default upsample
factors are `8 x 8 x 4 = 256`, exactly matching the current speech hop length.

The target-laptop CPU benchmark passed:

- generator parameters: 283,425
- exact output length: 256 waveform samples per mel frame
- 8 bounded optimization steps: loss 0.436857 -> 0.269484
- mean training step: 0.0395 seconds
- 1.024 s waveform inference median: 0.009 s
- median real-time factor: 0.0088
- median real-time multiple: 113.28x

This establishes local compute feasibility only. It does not establish naturalness,
identity fidelity, or final waveform quality.

## Persistent training contract

Before any long vocoder run, LYKENOX now uses an explicit versioned training boundary:

- `vocoder-segment-v1`: deterministic train/validation mel-wave pairing with exact
  hop-aligned sample lengths and a stable segment seed
- `vocoder-loss-v1`: small waveform L1 plus multi-resolution spectral convergence and
  log-magnitude reconstruction
- Stage A: reconstruction warm-up before adversarial pressure
- Stage B: reconstruction + lightweight hinge adversarial loss + discriminator feature
  matching
- LYKENOX-owned two-scale waveform discriminator implemented with generic PyTorch layers
- held-out validation reconstruction metric kept separate from the adversarial training
  objective
- checkpoint contains generator/discriminator states, both optimizer states, epoch/step,
  held-out metric, exact generator configuration, train/val manifest hashes, speech mel
  config/hash, segment contract version, loss recipe version, segment length and seed
- the final installed runtime requires only the generator

The persistent checkpoint's best-model rule is intentionally conservative: choose the
lowest finite held-out reconstruction loss while requiring finite adversarial metrics.
Perceptual listening on held-out reconstruction WAVs remains mandatory before the compact
architecture can be accepted for longer training.

## Contract gate

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_training_contract_smoke
```

This bounded smoke checks deterministic train/validation segmentation, exact waveform
length, reconstruction warm-up, discriminator and generator adversarial updates, feature
matching, finite gradients, held-out validation measurement, checkpoint provenance, both
optimizer states, and exact checkpoint round-trip.

The target laptop passed this gate with finite train/validation losses, exact checkpoint
round-trip, exact provenance and both optimizer states present.

## Persistent short-training gate

The next gate is the first run that is allowed to produce audio worth listening to, while
still remaining intentionally bounded. Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_short_train
```

Default contract:

- `vocoder-short-train-v1`
- 96 mel frames per segment (~1.024 s at 24 kHz / hop 256)
- 16 deterministic training segments
- 6 deterministic held-out validation segments
- up to 8 epochs
- 2 reconstruction-only warm-up epochs
- adversarial + feature matching afterward
- best checkpoint selected only by held-out reconstruction
- early stopping patience 3
- 85-second internal wall-clock budget
- compatible interrupted runs resume from `last.pt`

Artifacts are written under:

```text
models/lykenox_identity/training/vocoder_short_training/
```

The run writes `best.pt`, `last.pt`, `training_progress.json`, `training_report.json` and
up to three held-out `generated.wav` / `reference.wav` listening pairs from the best
checkpoint.

A numerical pass requires the best reloaded checkpoint to improve held-out reconstruction
over random initialization and reproduce its stored validation metric. A numerical pass is
**not** a perceptual pass. The listening pairs must be heard before any longer vocoder run.

## Gate order after the short run

1. Listen to the held-out generated/reference WAV pairs.
2. If the generated audio is dominated by noise, metallic artifacts, instability, or does
   not reconstruct recognizable speech structure, adjust or replace the current vocoder
   architecture/recipe before spending more CPU.
3. If reconstruction is recognizably speech-like and artifacts appear tractable, perform a
   controlled longer vocoder experiment with held-out validation and best-checkpoint
   selection.
4. Only after perceptual evidence should the project spend hours on long acoustic identity
   training and final end-to-end synthesis integration.
5. Export the eventual generator-only runtime artifact under the LYKENOX model manifest.
