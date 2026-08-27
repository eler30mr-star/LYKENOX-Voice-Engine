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

A pass advances to `persistent_vocoder_short_training`. That next run is still bounded:
it must create a best validation checkpoint and held-out WAV reconstructions for listening
before any long identity/acoustic training is allowed.
