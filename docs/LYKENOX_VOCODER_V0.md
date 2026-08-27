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

## v0 listening rejection

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
- v0 is not a candidate runtime vocoder

## Resize-convolution v1 rejection

`LykenoxVocoderGeneratorV1` removed every `ConvTranspose1d` stage and replaced learned
upsampling with linear interpolation followed by stride-1 convolutions. Its bounded probe
successfully removed the exact 93.75 Hz frame-rate lock, but the next held-out listening
and spectral gate still failed.

For the three generated validation WAVs examined after the v1 probe:

- no useful reconstructed speech was audible
- generated spectral centroids were only about 33 Hz, 116 Hz, and 56 Hz while references
  were roughly 1.8 kHz, 2.8 kHz, and 2.1 kHz
- more than 99.9% of generated spectral energy was below 80 Hz in all three examples
- energy above 300 Hz was effectively absent

This is a different failure from v0. The frame-rate buzz is gone, but linear interpolation
creates a smooth sample grid with no explicit learned sample-phase representation. More
v1 epochs are therefore blocked; the architecture needs a waveform-detail mechanism.

## Learned polyphase v2 corrective probe

`LykenoxVocoderGeneratorV2` uses learned 1-D polyphase/subpixel upsampling:

```text
mel-rate features
  -> stride-1 Conv1d predicts factor phase channels
  -> deterministic channel-to-time shuffle
  -> stride-1 refinement/residual blocks
  -> waveform
```

Properties:

- no `ConvTranspose1d`
- no interpolation upsampling bottleneck
- explicit learned sample-phase channels
- phase-equal initialization so the initial model does not begin with an arbitrary periodic
  phase pattern
- no per-phase expansion bias, removing another easy route to an unconditional carrier
- exact `8 x 8 x 4 = 256` sample expansion remains enforced
- CPU-bounded channel schedule for the target laptop

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_polyphase_probe
```

The v2 probe reports both known automatic failure modes:

- `frame_rate_lock_count_across_3_generated`
- `subbass_or_silence_collapse_count_across_3_generated`

A numeric pass requires held-out reconstruction improvement and zero detections for both
known artifact classes. Even then, the three generated/reference WAV pairs must be heard
before v2 can become the persistent vocoder architecture.
