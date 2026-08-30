from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.training.speech_vocoder_v4_2_level_attribution_audit import (
    AUDIT_VERSION,
    FULL_ROUTE_VARIANT,
    OUTPUT_DIR_NAME,
    TEACHER_GRID_VARIANTS,
    VALIDATION_INDICES,
    _active_rms,
    _reference_active_frame_mask,
    run_v4_2_level_attribution_audit,
)


class VocoderV42LevelAttributionContractTests(unittest.TestCase):
    def test_audit_identity_and_variants_are_fixed(self) -> None:
        self.assertEqual(AUDIT_VERSION, "vocoder-v4-2-level-attribution-audit-v1")
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        self.assertEqual(
            TEACHER_GRID_VARIANTS,
            (
                "v4_2_oracle",
                "v4_2_predicted_mel_target_prosody",
                "v4_2_target_mel_predicted_prosody",
                "v4_2_predicted_mel_predicted_prosody_teacher_grid",
            ),
        )
        self.assertEqual(FULL_ROUTE_VARIANT, "v4_2_full_reference_free")
        self.assertIn("level_attribution", OUTPUT_DIR_NAME)

    def test_audit_is_read_only_and_uses_required_guards(self) -> None:
        source = inspect.getsource(run_v4_2_level_attribution_audit)
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "run_bounded",
            "posthoc_gain_normalization_used\": True",
            "posthoc_eq_used\": True",
            "posthoc_denoising_used\": True",
            "training_authorized\": True",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("frame_grid_artifact", inspect.getsource(__import__(
            "lykenox_voice_engine.training.speech_vocoder_v4_2_level_attribution_audit",
            fromlist=["dummy"],
        )))
        self.assertIn('"checkpoints_unchanged": checkpoints_unchanged', source)
        self.assertIn('"training_started": False', source)
        self.assertIn('"training_authorized": False', source)
        self.assertIn('"predicted_duration_modified": False', source)

    def test_active_speech_rms_uses_reference_frame_mask(self) -> None:
        hop = 4
        waveform = torch.tensor(
            [
                0.0, 0.0, 0.0, 0.0,
                0.1, 0.1, 0.1, 0.1,
                1.0, 1.0, 1.0, 1.0,
                0.01, 0.01, 0.01, 0.01,
            ],
            dtype=torch.float32,
        )
        mask = _reference_active_frame_mask(waveform, hop)
        self.assertEqual(mask.tolist(), [False, True, True, False])
        expected = float(torch.sqrt(torch.tensor((4 * 0.01 + 4 * 1.0) / 8.0)))
        self.assertAlmostEqual(_active_rms(waveform, mask, hop), expected, places=6)


if __name__ == "__main__":
    unittest.main()
