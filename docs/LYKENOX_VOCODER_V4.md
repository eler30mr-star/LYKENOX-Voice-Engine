# LYKENOX Vocoder v4: pitch-conditioned source-filter

## Why the mel-only line is paused

Three different mel-only waveform paths exposed the same product risk from opposite
directions:

- v0 transposed-convolution: generated-specific carrier at `24000 / 256 = 93.75 Hz`
- v1 resize-convolution: carrier removed, but validation audio collapsed almost entirely
  into sub-bass and contained no useful speech reconstruction
- v2 learned polyphase: spectral capacity returned, but a generated-specific 93.75 Hz
  carrier was confirmed against references
- v3 smooth base + zero-mean gated phase residual + target-referenced hop loss: validation
  improved numerically, yet all three held-out generated examples still had the confirmed
  generated-specific frame lock

The v3 result closes an important hypothesis. Zero-mean phase channels do **not** prevent a
periodic AC carrier: a sinusoid can have zero mean. The v3 periodicity loss also measures
exact-hop excess correlation, which is weaker than explicitly controlling the physical
pitch source. Lower reconstruction loss is therefore not sufficient evidence that a
mel-only generator has learned the intended speech excitation.

No v0-v3 checkpoint is a runtime candidate.

## V4 architectural change

V4 changes the conditioning contract rather than adding another learned upsampler:

```text
mel + F0 + voicing
  -> deterministic sample-rate conditioning
  + harmonic excitation following F0
  + aperiodic excitation independent of mel hop
  -> LYKENOX depthwise-separable neural filter
  -> waveform
```

`LykenoxVocoderGeneratorV4` contains no `ConvTranspose1d` and no learned temporal
upsampling. Voiced periodicity comes from supplied F0, not from the 256-sample mel grid.
The neural network learns spectral shaping/timbre around that source.

For the architecture-selection probe, F0/voicing is extracted from the owned target WAV by
`lykenox-pitch-v1`, a deterministic PyTorch FFT-autocorrelation frontend. This is
**training supervision only**. The final installed speech product will not inspect a
reference WAV at inference. If v4 is accepted, the LYKENOX speech acoustic model must gain
its own predicted F0/voicing heads:

```text
text -> LYKENOX acoustic model -> mel + predicted F0 + predicted voicing
     -> LYKENOX v4 vocoder -> waveform
```

For future singing, score-conditioned pitch can feed the same vocoder interface directly.
This remains consistent with persistent identity synthesis: no source singer, voice
conversion, or inference-time reference recording is introduced.

## Bounded v4 gate

Run:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_source_filter_probe
```

The gate trains from scratch on deterministic real segments and reports:

- held-out reconstruction improvement
- confirmed generated-specific 93.75 Hz lock count
- sub-bass/silence collapse count
- generated/reference pitch diagnostics
- CPU update timing
- three generated/reference listening pairs

A numeric pass requires held-out reconstruction improvement plus zero known artifact
counts. Human listening is still required before any persistent v4 training contract is
created.
