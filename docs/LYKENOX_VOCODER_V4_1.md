# LYKENOX Vocoder v4.1 — source balance gate

## Current architectural decision

The vocoder remains a LYKENOX-owned, local source-filter model:

```text
LYKENOX acoustic outputs
  mel + F0 + voicing
        |
        v
LYKENOX pitch-conditioned source-filter vocoder
        |
        v
24 kHz waveform
```

This does **not** introduce reference-audio inference, voice conversion, a third-party TTS
backend, or an external vocoder. During the present vocoder-isolation experiments F0 and
voicing are extracted from owned target WAVs so the waveform stage can be tested with
correct conditioning. The final speech acoustic model must predict the same F0/voicing
contract from text/prosody; normal product inference will not inspect a reference WAV.

## Evidence accumulated before v4.1

The earlier compact mel-only generators were rejected before long training:

- v0 transposed-convolution: generated-specific carrier at exactly `24000 / 256 = 93.75 Hz`
- v1 resize-convolution: carrier removed, but generated energy collapsed almost entirely
  into sub-bass and no useful speech reconstruction emerged
- v2 free polyphase: spectral capacity returned, but the generated-specific 93.75 Hz carrier
  returned in all three held-out examples
- v3 constrained phase residual + target-referenced periodicity loss: still produced the
  confirmed carrier in all three held-out examples

V4 changed the conditioning contract instead of adding another upsampler. It uses explicit
F0/voicing excitation and a small neural source-filter. Its bounded CPU probe was the first
to clear both known structural gates:

- `confirmed_generated_specific_frame_locks: 0`
- `subbass_or_silence_collapse_count: 0`
- `automatic_artifact_gate_pass: true`
- 9,729 trainable parameters
- six-epoch probe completed in about 13 seconds on the target CPU

Human listening then showed meaningful improvement: held-out generated WAVs contained real
harmonic structure related to the references instead of only a hop carrier or sub-bass
collapse. They were not yet acceptable as a vocoder, however. The fundamental/lower bands
were too dominant and upper speech/formant energy remained weak. A generic pitch detector
also tended to select a strong higher harmonic (~300 Hz) even where spectral inspection
showed components near the intended ~95-102 Hz fundamental.

That result keeps v4 as the architecture family and motivates a local source-balance
correction rather than another architecture reset.

## v4.1 model changes

`LykenoxVocoderGeneratorV41` keeps the v4 contract and makes three bounded changes.

### 1. Mel-conditioned harmonic envelope

V4 used a permanently fixed source envelope:

```text
harmonic h amplitude = 1 / h
```

V4.1 predicts a bounded multiplicative deviation for each harmonic from the mel
conditioning. The final harmonic-envelope projection starts at zero, so initialization
reproduces the v4 `1/h` envelope exactly. Learned multipliers are bounded with `tanh` in log
space and the complete harmonic vector is RMS-normalized at every frame. This lets the
model redistribute energy from the fundamental into higher harmonics without winning the
loss simply by making the excitation louder.

### 2. DC/subgrave blocker below the speech F0 floor

A fixed 45 Hz linear-phase windowed-sinc high-pass FIR is applied before the final waveform
`tanh`. It is deliberately well below the pitch extractor's 60 Hz lower bound and is **not**
a notch at 93.75 Hz. Genuine low speech F0 near 80-100 Hz remains legal. The purpose is only
to prevent DC/subgrave drift from becoming an easy reconstruction shortcut.

The FIR is an owned deterministic runtime operation implemented with PyTorch `conv1d` and
ships as part of the generator state.

### 3. Target-relative broad-band balance loss

`vocoder-source-balance-v1` compares normalized generated/reference energy in:

```text
0-80 Hz
80-300 Hz
300-3000 Hz
3000-12000 Hz
```

The objective works on log band fractions and gives the 300-3000 Hz speech/formant region
the strongest weight. It is target-relative: a legitimately low-pitched reference is not
forced toward a generic spectral profile.

## Bounded v4.1 gate

Run only:

```powershell
.\.venv\Scripts\python.exe -m lykenox_voice_engine.training.speech_vocoder_source_balance_probe
```

Default probe contract:

- starts v4.1 from scratch; no v0-v4 checkpoint is resumed
- same 64-mel-frame (~0.683 s) segment size used by the v4 probe
- 12 deterministic train segments and 4 held-out validation segments
- 8 epochs total
- first 6 epochs: reconstruction + spectral-band balance
- final 2 epochs: same objectives plus deliberately mild adversarial/feature matching
- best checkpoint selected by held-out `reconstruction + 0.50 * spectral_balance`
- 85 second hard experiment budget

The automatic gate requires all of the following:

```text
validation_selection_improved: true
validation_spectral_balance_improved: true
confirmed_generated_specific_frame_locks: 0
subbass_or_silence_collapse_count: 0
upper_voice_band_missing_count: 0
automatic_artifact_gate_pass: true
```

`upper_voice_band_missing_count` is intentionally loose and target-relative. It only
rejects the previously observed near-empty >300 Hz failure mode; it is not a naturalness
metric.

The report also records, per listening pair:

- generated/reference broad-band fractions
- generated/reference legacy pitch-detector output
- periodic support at the *supplied* F0, which avoids automatically treating a selected
  second/third harmonic as proof that intended F0 is absent
- learned median harmonic weights
- frame-lock forensic data

## Acceptance rule

A v4.1 automatic pass is still **not** permission for long training. The three held-out
`generated.wav` files must be heard against their references.

V4.1 becomes the persistent-vocoder candidate only if the listening gate shows recognizable
speech structure in at least two of three held-out examples, with clearly stronger
mid/upper speech content than v4 and without reintroducing frame-grid buzz, silence/sub-bass
collapse, or an obviously wrong perceived pitch.

If that listening gate passes, the next engineering task is to convert the source-filter
experiment into a versioned persistent training/checkpoint contract and only then widen
data coverage/training duration. Separately, the speech acoustic model must gain LYKENOX
F0/voicing prediction heads before end-to-end text inference.
