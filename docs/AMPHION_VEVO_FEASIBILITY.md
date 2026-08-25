# Amphion Vevo2 / Vevo1.5 Feasibility for LYKENOX Direct Singing

Date: 2026-08-25
Project: D:\Proyectos\LYKENOX-Voice-Engine
Scope: audit only. No installs, no downloads, no app changes.

## Fixed Objective

Target pipeline:

```python
synthesize(
    lyrics="...",
    melody=score_or_prosody,
    timbre_ref="my_lykenox_voice.wav",
) -> singing.wav
```

The target is direct neural singing with the LYKENOX timbre. It must not depend on a mandatory source singer recording, RVC, or post-conversion of someone else's singing voice.

## Sources Checked

- Official repository: https://github.com/open-mmlab/Amphion
- Vevo2 official recipe: `models/svc/vevo2/README.md`, `infer_vevo2_ar.py`, `infer_vevo2_fm.py`, `requirements.txt`
- Vevo1.5 / VevoSing official recipe: `models/svc/vevosing/README.md`, `infer_vevosing_ar.py`, `infer_vevosing_fm.py`, `requirements.txt`
- Hugging Face checkpoints:
  - https://huggingface.co/RMSnow/Vevo2
  - https://huggingface.co/amphion/Vevo1.5

## Executive Conclusion

C) ninguno viable en esta laptop for the requested local microtest.

Reason: both Vevo2 and Vevo1.5 are conceptually close to the requested architecture, because their AR + FM paths can synthesize from text plus style/timbre reference and can use a prosody or melody reference. However, the released checkpoints are about 11 GB / 10.8 GB on disk, the full AR + FM stack loads multiple large models, and the practical path is designed for CUDA-class inference. They fall back to `torch.device("cpu")` in code, so CPU is not categorically impossible, but 12 GB system RAM is too tight for a reliable local run and inference would be extremely slow.

The second blocker is Spanish. The released model cards expose six languages, and the official Vevo1.5 training metadata lists `en`, `zh`, `ja`, `ko`, `fr`, and `de`. Spanish (`es`) is not listed. Adding Spanish without retraining the whole model is not confirmed by the official repo. A Spanish microtest would therefore be unsupported and likely fail in text/phoneme handling or pronunciation quality.

## Mode Separation

| Mode | Meaning | Fits LYKENOX Objective? |
| --- | --- | --- |
| Zero-shot text-to-singing | Generate singing from new text using reference audio for style/timbre. | Yes conceptually. |
| SVS | Singing voice synthesis from lyrics and singing prosody/melody conditioning. | Yes conceptually. |
| Melody control | Uses humming/piano/prosody audio as melody/prosody input. | Partially. It is audio/prosody-reference based, not a simple MIDI score API. |
| Timbre reference | `timbre_ref_wav_path` conditions identity/timbre. | Yes conceptually. |
| Style reference | `style_ref_wav_path` conditions style/prosody/expression. | Useful, but separate from identity. |
| SVC | Converts or edits an existing singing/speech audio source. | No as primary architecture. This is not the requested source-free path. |

## Real Inputs Observed

### Vevo1.5 AR + FM

Official function patterns include:

- `vevosing_tts(tgt_text, ref_wav_path, ref_text=None, timbre_ref_wav_path=None, src_language="en", ref_language="en")`
- `vevosing_melody_control(tgt_text, tgt_melody_wav_path, style_ref_wav_path, style_ref_text, timbre_ref_wav_path, tgt_language="en" or "zh")`

Real inputs:

- Text / lyrics: yes, `tgt_text` / `src_text`
- Language: yes, `src_text_language`, `style_ref_wav_text_language`
- Timbre reference WAV: yes, `timbre_ref_wav_path`
- Style reference WAV: yes, `style_ref_wav_path`
- Reference text: often used, `style_ref_wav_text` / `ref_text`
- Melody/prosody: yes, but as audio reference such as humming or piano WAV, not as a direct MIDI-score input in the sample API
- Mandatory source singer: no for `vevosing_tts`; yes for editing/SVC/conversion modes

Conceptual match:

```python
vevosing_melody_control(
    tgt_text="...",
    tgt_melody_wav_path="melody_or_humming.wav",
    style_ref_wav_path="my_voice_or_style.wav",
    timbre_ref_wav_path="my_voice.wav",
)
```

This is close, but not identical to `lyrics + MIDI melody + timbre_ref` because melody control expects a prosody/melody audio reference.

### Vevo2 AR + FM

Official function patterns include:

- `vevo2_tts(tgt_text, ref_wav_path, ref_text=None, timbre_ref_wav_path=None)`
- `vevo2_melody_control(tgt_text, tgt_melody_wav_path, style_ref_wav_path, style_ref_text, timbre_ref_wav_path)`

Real inputs:

- Text / lyrics: yes, `target_text`
- Language argument: not exposed in the simple Vevo2 sample functions
- Timbre reference WAV: yes, `timbre_ref_wav_path`
- Style reference WAV: yes, `style_ref_wav_path`
- Reference text: yes, `style_ref_wav_text`
- Melody/prosody: yes, `prosody_wav_path` / `tgt_melody_wav_path`
- Mandatory source singer: no for `vevo2_tts`; yes for editing/SVC/conversion modes

Conceptual match:

```python
vevo2_melody_control(
    tgt_text="...",
    tgt_melody_wav_path="melody_or_piano.wav",
    style_ref_wav_path="my_voice_style.wav",
    timbre_ref_wav_path="my_voice.wav",
)
```

Again, this is close to the LYKENOX objective, but the public sample is melody/prosody-audio based, not a direct score API.

## Spanish Support

Verdict: not supported as a confirmed released language.

Evidence:

- Vevo1.5 official training metadata says supported transcription languages are `en`, `zh`, `ja`, `ko`, `fr`, and `de`.
- Hugging Face cards show six languages for both Vevo1.5 and Vevo2.
- Spanish (`es`) is absent from the official listed languages.

Could Spanish be added without full retraining?

Unknown / not confirmed. A partial Spanish path might require at least text normalization/G2P/token mapping support and could still suffer because the AR model was not shown as trained for Spanish singing. The official docs do not provide a small Spanish adapter recipe. Treat Spanish as unsupported until proven by a live microtest or official update.

## Hardware Feasibility

Target hardware:

- Windows
- Intel i7-1255U
- 12 GB RAM
- Intel Iris Xe
- no CUDA
- no working XPU path

### Code-level device behavior

Both AR inference scripts choose:

```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

So the code is not obviously CUDA-only at entry point. Dependencies include regular PyTorch, transformers, tokenizers/vocoders, and audio packages. Vevo2 additionally depends on `torchcrepe`, which can run on CPU in some environments but is normally heavy.

### Practical CPU/RAM assessment

| Item | Vevo2 | Vevo1.5 |
| --- | --- | --- |
| Checkpoint repo size | about 11 GB | about 10.8 GB |
| AR model | Qwen2.5-0.5B post-trained | AR Transformer 780M |
| FM model | 350M | 350M |
| Vocoder | 250M | 250M |
| Tokenizers | prosody + content-style | prosody + content-style |
| Expected RAM for full AR + FM CPU | likely >12 GB once models, tensors, Python, audio buffers, and temporary activations are loaded | likely >12 GB once full stack is loaded |
| CPU inference | technically possible by code fallback, not practically suitable here | technically possible by code fallback, not practically suitable here |
| 5-10 s estimate on i7-1255U CPU | likely tens of minutes or may fail with RAM pressure | likely tens of minutes or may fail with RAM pressure |
| CUDA required | not at script entry, but expected for practical inference | not at script entry, but expected for practical inference |
| Windows | not officially impossible, but recipes are Linux/conda oriented; espeak-ng and audio deps add friction | same, plus phonemizer/espeak-ng requirement |

Conclusion: Do not choose either for a local CPU/12 GB microtest unless the goal is only environment probing, not generation quality.

## Checkpoints Needed for SVS / Text-to-Singing

### Vevo2

All are needed for AR + FM text-to-singing / SVS:

- `tokenizer/prosody_fvq512_6.25hz`
- `tokenizer/contentstyle_fvq16384_12.5hz`
- `contentstyle_modeling/posttrained`
- `acoustic_modeling/fm_emilia101k_singnet7k_repa`
- `vocoder`

Total advertised repository size: about 11 GB.

### Vevo1.5

All are needed for AR + FM text-to-singing / SVS:

- `tokenizer/prosody_fvq512_6.25hz`
- `tokenizer/contentstyle_fvq16384_12.5hz`
- `contentstyle_modeling/ar_emilia101k_singnet7k`
- `acoustic_modeling/fm_emilia101k_singnet7k`
- `acoustic_modeling/Vocoder`

Total advertised repository size: about 10.8 GB.

FM-only modes do not satisfy the LYKENOX objective because they are mainly VC/SVC/timbre conversion paths that require source audio content.

## Training / Adaptation

Preferred path: no training, use timbre zero-shot from `timbre_ref_wav_path`.

Both families claim zero-shot voice imitation and expose timbre reference inputs. Therefore, if language and hardware were acceptable, the first experiment would be zero-shot only, no speaker adaptation.

No official small speaker-embedding or lightweight adapter recipe was found for adapting one Spanish LYKENOX voice on CPU. Full training recipes involve tokenizers, AR model, FM model, and vocoder training with large speech/singing datasets. That is not viable on this laptop.

## Vevo2 vs Vevo1.5

| Characteristic | Vevo2 | Vevo1.5 / VevoSing |
| --- | --- | --- |
| Text-to-singing | Yes in official task list and AR sample | Yes in official task list and AR sample |
| Melody control | Yes, via prosody/melody audio such as humming/piano | Yes, via prosody/melody audio such as humming/piano |
| Timbre reference | Yes, `timbre_ref_wav_path` | Yes, `timbre_ref_wav_path` |
| Zero-shot identity | Yes conceptually | Yes conceptually |
| Style reference | Yes | Yes |
| SVC mode | Yes, but not our target | Yes, but not our target |
| Languages | HF says 6 languages; Spanish not listed | Official list: en, zh, ja, ko, fr, de; Spanish not listed |
| Direct MIDI/score API | Not shown in official samples | Not shown in official samples |
| CPU path | Code falls back to CPU | Code falls back to CPU |
| Practical CPU on 12 GB | No | No |
| Checkpoint size | about 11 GB | about 10.8 GB |
| Windows complexity | High | High, plus phonemizer/espeak-ng friction |
| Maturity for microtest | Newer, simpler full repo download, but still heavy | More explicit language args and samples, but still heavy |

## Microtest Choice

If ignoring laptop limits and Spanish, choose Vevo2 for a future GPU/server microtest because it is the newer unified model, has a simpler single checkpoint repo layout, and exposes direct `vevo2_melody_control` with text, melody/prosody WAV, style reference, and timbre reference.

For this laptop and this Spanish LYKENOX objective: choose none.

## Required Final Answers

- Vevo2 viable for microtest local: No.
- Vevo1.5 viable for microtest local: No.
- Ninguno viable en esta laptop: Yes.
- Can use my timbre without training: Conceptually yes via `timbre_ref_wav_path`; practically untested locally.
- Can sing new text: Yes for supported languages in AR + FM mode.
- Can control melody: Yes via prosody/melody audio reference, not confirmed as direct MIDI score.
- Supports Spanish: No confirmed support.
- Requires source singer: No for TTS/text-to-singing; yes for editing/SVC/conversion modes.
- RAM: 12 GB is likely insufficient for full AR + FM inference reliably.
- Disk: about 11 GB Vevo2, about 10.8 GB Vevo1.5, before env/cache overhead.
- CPU: code fallback exists, but practical inference is not recommended.
- Estimated time for 5-10 s on CPU: likely tens of minutes or failure under RAM pressure.

## Decision

Do not integrate Amphion Vevo2 or Vevo1.5 into LYKENOX on this laptop now. Keep them as future candidates only if a CUDA GPU or remote GPU runner is available and if Spanish support is either officially added or a controlled Spanish phoneme/text path is validated.
