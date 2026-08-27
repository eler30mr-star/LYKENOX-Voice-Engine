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
inside this repository. It uses a mel pre-convolution, three learned transposed-convolution
upsampling stages, local residual convolution blocks, and a bounded waveform output. The
default upsample factors are `8 x 8 x 4 = 256`, exactly matching the current speech hop
length.

The target-laptop CPU benchmark passed:

- generator parameters: 283,425
- exact output length: 256 waveform samples per mel frame
- 8 bounded optimization steps: loss 0.436857 -> 0.269484
- mean training step: 0.0395 seconds
- 1.024 s waveform inference median: 0.009 s
- median real-time factor: 0.0088
- median real-time multiple: 113.28x

This established local compute feasibility only. It did not establish naturalness,
identity fidelity, or final waveform quality.

## Persistent training contract

Before any long vocoder run, LYKENOX uses an explicit versioned training boundary:

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
Perceptual listening on held-out reconstruction WAVs remains mandatory before an
architecture can be accepted for longer training.

## Contract gate

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_training_contract_smoke
```

The contract smoke passed, and the subsequent bounded persistent short training also
passed numerically. Its best checkpoint improved held-out reconstruction, but the first
human listening gate rejected the transposed-convolution v0 output: generated validation
WAVs contained a strong nearly fixed periodic sound instead of reconstructed speech.

Offline forensic analysis of the three held-out generated WAVs found their dominant pitch
locked almost exactly to `24000 / 256 = 93.75 Hz`, the mel-frame rate, while the reference
speech pitch varied normally. The generated spectral fingerprints were also unusually
similar across different validation segments. This is treated as an architectural
upsampling artifact, not as evidence that more v0 epochs are warranted.

Therefore:

- do not continue long training of `lykenox_compact_transposed_conv_v0`
- preserve the v0 checkpoints only as diagnostic history
- test the v1 resize-convolution replacement before further vocoder investment

## Resize-convolution v1 corrective probe

`LykenoxVocoderGeneratorV1` removes every `ConvTranspose1d` stage. Each upsampling stage
uses deterministic linear interpolation followed by stride-1 `Conv1d` refinement and the
same compact residual concept. The audio/mel contract remains exactly 24 kHz and 256
samples per mel frame.

Run the bounded architecture-selection probe:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_resizeconv_probe
```

It trains v1 briefly on the same deterministic mel/wave contract, selects the best held-out
reconstruction epoch, writes three generated/reference WAV pairs, and reports a simple
frame-rate-lock diagnostic. A numeric pass alone does not accept v1; the generated WAVs
must contain recognizable speech structure and must not reproduce the fixed 93.75 Hz buzz.
