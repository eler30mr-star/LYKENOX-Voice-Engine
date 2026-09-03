from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_candidate_magnitude_phase_recovery_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_candidate_magnitude_phase_recovery_v1", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class CandidateMagnitudePhaseRecoveryDiagnosticTests(unittest.TestCase):
    def test_scope_is_fixed_to_two_accepted_forensic_items(self) -> None:
        self.assertEqual(
            diagnostic.DEFAULT_UTTERANCE_IDS,
            (
                "speech_0021_6cd35984e877_seg_001",
                "speech_0022_ba721f6129b9_seg_005",
            ),
        )
        self.assertEqual(diagnostic.GRIFFIN_LIM_ITERATIONS, 64)
        self.assertEqual(diagnostic.GRIFFIN_LIM_SEED, 20260903)

    def test_phase_recovery_uses_candidate_magnitude_not_target_phase(self) -> None:
        source = inspect.getsource(diagnostic.run_candidate_magnitude_phase_recovery)
        griffin = inspect.getsource(diagnostic._griffin_lim_from_magnitude)
        self.assertIn("candidate_spec.abs()", source)
        self.assertIn("_griffin_lim_from_magnitude", source)
        self.assertIn("torch.istft", griffin)
        self.assertIn("torch.stft", inspect.getsource(diagnostic._stft))
        self.assertNotIn("torch.angle(target_spec)", griffin)

    def test_known_good_target_phase_hybrid_is_only_a_listening_ceiling(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertIn("candidate_mag_target_phase_ceiling", source)
        self.assertIn("candidate_mag_griffinlim64_render", source)
        self.assertIn("target_mag_griffinlim64_render", source)

    def test_no_training_renderer_edit_or_postprocess_path(self) -> None:
        lowered = inspect.getsource(diagnostic).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "normalize(",
            "equalizer",
            "denoise",
            "from_pretrained",
        ):
            if forbidden == "denoise":
                self.assertIn('"posthoc_denoising_used": false', lowered)
            else:
                self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
