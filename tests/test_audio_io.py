from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from lykenox_voice_engine.audio.io import load_audio


class AudioIoTests(unittest.TestCase):
    def test_load_audio_reads_pcm_wav_without_torchcodec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            sample_rate = 24000
            samples = np.zeros((sample_rate // 10,), dtype=np.float32)
            sf.write(path, samples, sample_rate, subtype="PCM_16")

            waveform, loaded_rate = load_audio(path)

            self.assertEqual(loaded_rate, sample_rate)
            self.assertEqual(tuple(waveform.shape), (1, len(samples)))
            self.assertEqual(str(waveform.dtype), "torch.float32")


if __name__ == "__main__":
    unittest.main()
