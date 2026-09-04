from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_target_phase_magnitude_band_cleanup_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_target_phase_magnitude_band_cleanup_v1",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class TargetPhaseMagnitudeBandCleanupDiagnosticTests(unittest.TestCase):
    def test_gate_is_narrow_and_no_training(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertEqual(diagnostic.LOW_BAND_MAX_HZ, 1800.0)
        self.assertEqual(diagnostic.MID_BAND_MAX_HZ, 4500.0)
        self.assertIn('"training_executed": False', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"checkpoint_written": False', source)
        self.assertIn('"renderer_modified": False', source)
        self.assertIn('"product_posthoc_gain_normalization_used": False', source)
        self.assertIn('"audition_monitor_gain_used": True', source)
        self.assertIn('"audition_monitor_gain_common_within_each_utterance": True', source)
        self.assertIn('"posthoc_eq_used": False', source)
        self.assertIn('"posthoc_denoising_used": False', source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("torch.save(", source)

    def test_only_magnitude_band_changes_under_target_phase(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertIn("target_phase = torch.angle(target_spec)", source)
        self.assertIn("target_band_mask=low_mask", source)
        self.assertIn("target_band_mask=mid_mask", source)
        self.assertIn("target_band_mask=high_mask", source)
        self.assertIn("target_band_mask=all_mask", source)
        self.assertIn("torch.where(", source)
        self.assertIn("torch.polar(magnitude.to(torch.float32), target_phase.to(torch.float32))", source)

    def test_raw_and_audition_outputs_are_separate(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertIn('raw_dir = output_dir / "raw"', source)
        self.assertIn('audition_dir = output_dir / "audition"', source)
        self.assertIn('__AUDITION.wav', source)
        self.assertIn("renders[key] * audition_gain", source)


if __name__ == "__main__":
    unittest.main()
