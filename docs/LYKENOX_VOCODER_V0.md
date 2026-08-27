# LYKENOX Vocoder v0

## Product boundary

The final speech product requires an owned local waveform stage:

```text
LYKENOX acoustic mel -> LYKENOX vocoder -> 24 kHz waveform
```

The installed product must not require an external vocoder executable, hosted API,
reference speaker, source voice, or model download during normal inference.

## First CPU feasibility architecture

`LykenoxVocoderGenerator` is a compact non-autoregressive PyTorch generator implemented
inside this repository. It uses a mel pre-convolution, three learned upsampling stages,
local residual convolution blocks, and a bounded waveform output. The default upsample
factors are `8 x 8 x 4 = 256`, exactly matching the current speech hop length.

This v0 generator is a compute probe. It is not yet the final perceptually trained
vocoder architecture and no quality claim should be made from the benchmark alone.

## Benchmark gate

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_cpu_benchmark
```

The bounded gate uses real cached mel features and the corresponding local WAV. It checks:

- exact 256 waveform samples per mel frame
- finite generator forward/backward on CPU
- short optimization loss decreases
- gradient norms remain finite
- inference timing on roughly one second of generated 24 kHz audio
- median real-time factor <= 1.0 for the current prototype

A pass establishes local compute feasibility only. It does not establish naturalness,
identity fidelity, or final waveform quality.

## Gate order after a pass

1. Define persistent vocoder training/validation checkpoints and loss contract.
2. Benchmark the selected training objective and memory/time bounds before a long run.
3. Train a bounded experimental vocoder and listen to held-out reconstructions.
4. Only after perceptual evidence, decide whether the architecture is adequate or must
   be replaced before long identity training.
5. Export the eventual runtime artifact under the LYKENOX model manifest.
