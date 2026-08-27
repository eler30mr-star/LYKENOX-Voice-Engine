# LYKENOX Product Independence

## Product rule

LYKENOX Voice Engine is the product. It must not require another TTS/SVS product,
cloud API, source voice, source singer, or reference-audio prompt in normal inference.

Final contracts:

```text
text -> LYKENOX speech model -> speech.wav
lyrics + score -> LYKENOX singing model -> singing.wav
```

The installed application must continue to synthesize offline from packaged model
artifacts even if an upstream research repository disappears.

## What is allowed

LYKENOX may use redistributable infrastructure libraries such as numerical runtimes,
audio codecs, and ONNX Runtime, subject to their licenses. Published neural
architectures and research papers may be implemented inside LYKENOX. External projects
may be studied during R&D or used as temporary training references when legally
permitted.

The final inference artifact, API, model manifest, dataset format, application UI, and
runtime contract belong to LYKENOX and must not depend on a third-party TTS executable.

## What is not allowed as the product runtime

- `piper.exe` or Piper Python package as a required runtime engine
- Coqui TTS as a required runtime engine
- OpenUtau as a required application/runtime
- an external hosted API
- a model that needs a reference WAV on every request
- RVC/SVC/post-conversion as the primary identity path
- a source speaker or source singer at inference time

WORLDLINE-R remains an explicitly non-final research/fallback renderer for the existing
sample voicebank.

## Ownership boundary

### LYKENOX-owned

- master identity dataset schema and metadata
- Spanish text/singing frontend contract
- `/speak` and `/sing` API contracts
- model manifest and artifact layout
- training orchestration and validation
- inference runtime interfaces
- desktop product and installer
- trained LYKENOX identity artifacts produced from the owner's recordings

### Replaceable implementation details

- neural architecture family
- optimizer/training recipe
- tensor runtime
- vocoder architecture
- research code used to validate an approach

These details may change without changing the LYKENOX public API or dataset.

## Model artifact layout

Target layout:

```text
models/lykenox_identity/
  manifest.json
  speech/
    model.onnx              # or a future LYKENOX-native artifact
    frontend.json
    vocab.json
  singing/
    model.onnx              # or a future LYKENOX-native artifact
    frontend.json
    phonemes.json
```

The manifest records architecture provenance for reproducibility, but provenance does
not create a runtime dependency on that upstream project.

## Dataset rule

The master dataset is engine-neutral. Trainer-specific folders are generated views and
must be reproducible from the master metadata. Never make Piper, StyleTTS2, DiffSinger,
or another project's metadata format the canonical source of the LYKENOX identity.

## Current decision

Do not integrate Piper as the product backend. Piper may be studied as a reference for
compact CPU-oriented VITS inference/training decisions. Any speech model selected for
production must be loaded directly by a LYKENOX runtime through a self-contained model
artifact.

The same rule applies to the future neural singing model.
