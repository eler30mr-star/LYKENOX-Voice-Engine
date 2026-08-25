# Singing Voice Synthesis Engine Audit

Hardware target: Intel Core i7-1255U, Intel Iris Xe, 12 GB RAM, Windows, no CUDA, no verified XPU.

| Candidate | License | Windows | CPU | Training | Inference | Speaker identity | Input | Output | CUDA need | Maintenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NNSVS / ENUNU | MIT | Possible, setup can be strict | Possible but slow | Own voicebank training possible | Local | Own singer model | score labels / phonemes / pitch / durations | waveform via vocoder | Not mandatory for tiny tests; useful for real training | Stable research/community |
| DiffSinger / OpenVPI ecosystem | MIT-style code; weights vary | Possible but setup-heavy | Slow | Training is GPU-oriented | Local | Voice model training/fine-tuning | lyrics plus MIDI/F0/timing | waveform via vocoder | Strongly recommended | Active community |
| OpenUtau + ENUNU/DiffSinger | MIT | Yes | Backend-dependent | Delegates to backend | Local editor/render shell | Voicebank-dependent | UST/MIDI-style notes/lyrics | rendered audio | Backend-dependent | Active |
| ESPnet / MUSKIT SVS | Apache-2.0 | Possible, Linux-like tooling preferred | Slow | Research training recipes | Local | Multi-speaker recipes exist | score/phoneme/duration/F0 | waveform | Expected for training | Active research |

## Recommendation

Recommended Phase 2 candidate: NNSVS/ENUNU-class backend for the first local microtest.

Reason: it is a true score-to-singing path, supports creating your own voicebank, maps naturally
to lyrics plus notes plus timing, and has the most plausible CPU-only microtest path. The tradeoff
is quality. DiffSinger/OpenVPI is the higher-quality route, but on this machine training should be
treated as GPU-required unless an isolated XPU runtime is proven.

## Training vs Inference

- Training: possible only as microtests on CPU. Full-quality training is not practical on this hardware.
- Inference: possible locally if a compatible trained model and vocoder exist, but CPU latency may be high.
- Phase 1 installs no backend and produces no fake vocals.
