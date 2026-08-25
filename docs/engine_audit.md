# Singing Voice Synthesis Engine Audit

| Motor | Licencia | Singing synthesis | Speaker identity | Fine-tuning | CPU training | CPU inference | Windows | Input musical | RAM/tamaño | Estado |
|---|---|---|---|---|---|---|---|---|---|---|
| DiffSinger / OpenVPI ecosystem | MIT for original code; vocoders/voicebanks vary | Yes | Voicebank/model identity | Yes, but dataset labeling/training toolchain required | Not practical | Possible for rendering, often via OpenUtau CPU/DirectML | Community Windows workflows exist | phoneme/lyrics + pitch/duration, often via UST/OpenUtau/MIDI-like score | Vocoder + acoustic models; training expects GPU | Most aligned, but training on CPU laptop is risky |
| NNSVS | BSD-style project ecosystem | Yes | Singer-specific models | Yes via recipes | Possible in theory, GPU strongly recommended | Yes | Windows tested, Linux preferred for development | score labels, pitch, timings | Smaller research stack, compiler needed | Technically clean but more engineering-heavy |
| ESPnet/Muskits SVS | Apache-2.0 | Yes | Multi-speaker/singer embeddings | Yes, research recipes | Not practical | Possible but heavy | Windows not the primary path | score lyrics + duration + pitch | Large toolkit | Strong research toolkit, too heavy for phase 1 |
| Coqui/XTTS-style TTS | MPL/varies | No, TTS not SVS | Speaker cloning | Yes/limited | CPU training not practical | Yes for speech | Windows possible | text only, no score-to-singing | Moderate-large | Not suitable as primary singing engine |

## Recommendation

Recommended first backend target: NNSVS for a strict local score-to-singing architecture, or DiffSinger/OpenVPI if the priority is compatibility with existing singer voicebank workflows. For this laptop, training a high-quality identity model locally on CPU is not realistic. Inference/rendering is more realistic than training.

## Next step

Prototype the engine adapter against NNSVS install/import and a tiny synthetic score fixture, without training a real voice model yet.
