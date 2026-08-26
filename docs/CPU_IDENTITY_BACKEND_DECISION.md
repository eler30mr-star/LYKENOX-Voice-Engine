# CPU Identity Backend Decision

Date: 2026-08-26

## Objective

Build LYKENOX as a direct identity voice engine:

```text
text -> trained LYKENOX identity model -> speech.wav
lyrics + melody -> trained LYKENOX identity model -> singing.wav
```

This project must not use a source speaker/singer and must not convert another
voice into LYKENOX.

## Current Machine Constraint

- Windows laptop
- CPU-only for training and inference
- No CUDA
- 12 GB RAM class machine

## Rejected Route

Coqui VITS from scratch is rejected.

Evidence:

- Local training ran, but generated unusable/silent output.
- `outputs/identity_epoch1/speech_epoch1.wav` measured as invalid/silence.
- Continuing that route would waste time and would not be professional.

## Backend Reality Check

### Piper

Piper is strong for CPU inference after a voice exists, but its official training
guide centers on dataset preparation, VITS training/fine-tuning, and checkpoint
export. The guide recommends fine-tuning existing checkpoints and shows GPU-based
training examples. It is not a practical from-scratch CPU training answer for
this laptop.

Verdict: not the next local CPU training route.

### Mimic 3

Mimic 3 is designed for local CPU inference and can run on low-end hardware, but
the project guidance for creating a custom voice warns that building a custom TTS
voice is a significant time investment and historically needs many hours of
consistent, high-quality audio.

Verdict: good CPU inference class, not a quick local custom-voice training route.

### Coqui Voice Cloning Models

Coqui supports voice cloning models where a reference WAV conditions generation.
That is not the selected objective here because the user does not want a voice
cloning/conversion-like path presented as the main architecture.

Verdict: do not use as the main LYKENOX path.

## Professional Decision

Do not continue neural-from-scratch TTS/SVS training on this CPU laptop as the
main route.

For this machine, the only honest local production path right now is:

1. Keep direct API contracts for `/speak` and `/sing` failing until a real model
   exists.
2. Preserve and improve the user's own dataset.
3. Use local sample-based singing only where it is explicitly sample-based and
   not mislabeled as neural identity.
4. If neural identity remains mandatory, train on suitable hardware or use a
   backend that explicitly supports CPU-friendly personal speaker adaptation
   without source voice conversion. No such backend is currently validated here.

## Immediate Engineering Rule

No more backend installs or training runs until the backend is proven to satisfy:

- direct text-to-speech with LYKENOX identity
- no source speaker at inference
- no RVC/SVC/voice conversion
- Spanish support
- CPU feasibility on this machine, or an explicit statement that training must
  happen off-machine
