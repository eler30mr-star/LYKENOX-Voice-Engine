from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import torch

from lykenox_voice_engine.training import speech_band_aperiodicity_calibration as band_cal
from lykenox_voice_engine.training import speech_glottal_calibration as glottal_cal
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_glottal_excitation_v1 as excitation
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as renderer


ORACLE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "diagnostic_calibrated_glottal_oracle_v1.py"
)
SPEC = importlib.util.spec_from_file_location("diagnostic_calibrated_glottal_oracle_v1", ORACLE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {ORACLE_SCRIPT}")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def _summary(value: float) -> dict[str, float | int]:
    return {"count": 10, "median": value, "min": value, "max": value, "p10": value, "p90": value}


def _write_calibrations(root: Path, *, owned: bool = True) -> None:
    target = root / "models" / "lykenox_identity" / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    glottal_bins = []
    for center, oq, asym, tilt, rms in (
        (100.0, 0.58, 0.31, -10.0, 0.20),
        (200.0, 0.52, 0.27, -12.0, 0.18),
    ):
        glottal_bins.append(
            {
                "f0_center_hz": center,
                "open_quotient": _summary(oq),
                "asymmetry_peak_position": _summary(asym),
                "spectral_tilt_db_per_octave": _summary(tilt),
                "residual_rms": _summary(rms),
            }
        )
    glottal = {
        "calibration_version": glottal_cal.GLOTTAL_CALIBRATION_VERSION,
        "policy_id": "LYX-POL-001",
        "split": "train",
        "owned_data_only": owned,
        "third_party_model_or_checkpoint_used": False,
        "sample_rate": 24000,
        "global": {
            "open_quotient": _summary(0.55),
            "asymmetry_peak_position": _summary(0.29),
            "spectral_tilt_db_per_octave": _summary(-11.0),
            "residual_rms": _summary(0.19),
        },
        "f0_bins": glottal_bins,
    }
    band_keys = [f"{int(low)}_{int(high)}_hz" for low, high in band_cal.BANDS_HZ]
    band_bins = []
    for center, values in (
        (100.0, (0.05, 0.10, 0.22, 0.42)),
        (200.0, (0.07, 0.14, 0.28, 0.50)),
    ):
        band_bins.append(
            {
                "f0_center_hz": center,
                "bands": {key: _summary(value) for key, value in zip(band_keys, values)},
            }
        )
    band = {
        "calibration_version": band_cal.BAND_APERIODICITY_CALIBRATION_VERSION,
        "policy_id": "LYX-POL-001",
        "split": "train",
        "owned_data_only": owned,
        "third_party_model_or_checkpoint_used": False,
        "sample_rate": 24000,
        "bands_hz": [
            {"low_hz": low, "high_hz": high, "key": key}
            for (low, high), key in zip(band_cal.BANDS_HZ, band_keys)
        ],
        "global": {
            "bands": {
                key: _summary(value)
                for key, value in zip(band_keys, (0.06, 0.12, 0.25, 0.46))
            }
        },
        "f0_bins": band_bins,
    }
    (target / "glottal_pulse_v1.json").write_text(json.dumps(glottal), encoding="utf-8")
    (target / "band_aperiodicity_v1.json").write_text(json.dumps(band), encoding="utf-8")


class GlottalCalibrationContractTests(unittest.TestCase):
    def test_calibrations_are_owned_cpu_measurements_not_training(self) -> None:
        glottal_source = inspect.getsource(glottal_cal)
        band_source = inspect.getsource(band_cal)
        for source in (glottal_source, band_source):
            lowered = source.lower()
            self.assertIn('"owned_data_only": True', source)
            self.assertIn('"training_executed": False', source)
            self.assertIn('"third_party_model_or_checkpoint_used": False', source)
            self.assertNotIn("torch.optim", lowered)
            self.assertNotIn(".backward(", lowered)
            self.assertNotIn("from_pretrained", lowered)
            self.assertNotIn("cuda", lowered)
        self.assertEqual(glottal_cal.DEFAULT_SPLIT, "train")
        self.assertEqual(band_cal.DEFAULT_SPLIT, "train")
        self.assertIn("wav_sha256", glottal_source)
        self.assertIn("wav_sha256", band_source)

    def test_real_residual_method_matches_step3f_geometry(self) -> None:
        frames = 10
        samples = frames * renderer.HOP_LENGTH
        index = torch.arange(samples, dtype=torch.float32)
        waveform = 0.2 * torch.sin(index * 0.053) + 0.1 * torch.sin(index * 0.111)
        residual, cepstrum, extension = glottal_cal.extract_owned_real_residual(
            waveform,
            frame_count=frames,
        )
        self.assertEqual(residual.shape, waveform.shape)
        self.assertEqual(cepstrum.shape, (frames, renderer.CEPSTRAL_ORDER))
        self.assertEqual(extension, 1)
        self.assertTrue(bool(torch.isfinite(residual).all()))


class CalibratedExcitationContractTests(unittest.TestCase):
    def test_candidate_requires_owned_calibration_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_calibrations(root, owned=False)
            with self.assertRaises(ValueError):
                excitation.OwnedCalibratedGlottalExcitationV1.from_root(root)

    def test_candidate_is_deterministic_and_exact_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_calibrations(root)
            model = excitation.OwnedCalibratedGlottalExcitationV1.from_root(root)
            frames = 12
            f0 = torch.linspace(110.0, 180.0, frames, dtype=torch.float32).unsqueeze(0)
            voiced = torch.ones_like(f0)
            periodicity = torch.full_like(f0, 0.82)
            first = model.build(f0, voiced, periodicity, noise_seed=17)
            repeat = model.build(f0, voiced, periodicity, noise_seed=17)
            different = model.build(f0, voiced, periodicity, noise_seed=18)
            self.assertEqual(first.shape, (1, frames * renderer.HOP_LENGTH))
            self.assertTrue(torch.equal(first, repeat))
            self.assertFalse(torch.equal(first, different))
            self.assertTrue(bool(torch.isfinite(first).all()))

    def test_filter_bank_is_complementary(self) -> None:
        kernels = excitation._complementary_band_kernels(
            device=torch.device("cpu"),
            dtype=torch.float64,
            sample_rate=renderer.SAMPLE_RATE,
        )
        self.assertEqual(len(kernels), 5)
        combined = torch.stack(kernels, dim=0).sum(dim=0)
        expected = torch.zeros_like(combined)
        expected[(excitation.FILTER_TAPS - 1) // 2] = 1.0
        self.assertLess(float((combined - expected).abs().max()), 1.0e-12)

    def test_candidate_is_rosenberg_and_uses_calibrated_band_aperiodicity(self) -> None:
        source = inspect.getsource(excitation)
        self.assertIn("_rosenberg_cycle", source)
        self.assertIn("open_quotient", source)
        self.assertIn("asymmetry_peak_position", source)
        self.assertIn("target_tilt_db_per_octave", source)
        self.assertIn("residual_rms", source)
        self.assertIn("band_aperiodicity", source)
        self.assertIn("torch.Generator(device=\"cpu\")", source)
        lowered = source.lower()
        for forbidden in ("torch.optim", ".backward(", "from_pretrained", "vocos", "hifigan", "bigvgan", "cuda"):
            self.assertNotIn(forbidden, lowered)


class CalibratedGlottalOracleContractTests(unittest.TestCase):
    def test_oracle_changes_only_excitation_and_requires_listening(self) -> None:
        source = inspect.getsource(oracle)
        self.assertEqual(oracle.DEFAULT_SPLIT, "val")
        self.assertEqual(oracle.DEFAULT_ITEMS, 3)
        self.assertIn("OwnedCalibratedGlottalExcitationV1.from_root", source)
        self.assertIn("reference_log_magnitude_to_one_sided_cepstrum", source)
        self.assertIn("render_time_varying_minimum_phase", source)
        self.assertIn('"envelope_filter_path_changed": False', source)
        self.assertIn('"synthetic_excitation_path_changed": True', source)
        self.assertIn('"production_renderer_modified": False', source)
        self.assertIn('"metrics_can_accept_product_quality": False', source)
        self.assertIn('"human_full_utterance_listening_required": True', source)
        self.assertIn("real_residual_resynthesis_ceiling", source)
        self.assertIn("original_synthetic_excitation_oracle", source)
        lowered = source.lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "cuda",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
