from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_renderer as production


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostic_real_residual_resynthesis_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_real_residual_resynthesis_v1", SCRIPT_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class RealResidualResynthesisDiagnosticTests(unittest.TestCase):
    def test_complex_transfer_matches_official_minimum_phase_fir(self) -> None:
        torch.manual_seed(11)
        frames = 4
        cepstrum = torch.randn(
            frames,
            production.CEPSTRAL_ORDER,
            dtype=torch.float64,
        ) * 0.01
        transfer = diagnostic._minimum_phase_transfer_from_cepstrum(cepstrum)
        impulse_from_transfer = torch.fft.irfft(
            transfer,
            n=production.N_FFT,
            dim=-1,
        )
        official = production.one_sided_real_cepstrum_to_minimum_phase_fir(
            cepstrum,
            n_fft=production.N_FFT,
        )
        self.assertEqual(impulse_from_transfer.shape, official.shape)
        self.assertLess(float((impulse_from_transfer - official).abs().max()), 1.0e-12)

    def test_identity_filter_residual_roundtrip_preserves_waveform(self) -> None:
        frames = 12
        samples = frames * production.HOP_LENGTH
        index = torch.arange(samples, dtype=torch.float32)
        waveform = (
            0.3 * torch.sin(index * 0.041)
            + 0.15 * torch.sin(index * 0.097)
        )
        cepstrum = torch.zeros(
            frames,
            production.CEPSTRAL_ORDER,
            dtype=torch.float32,
        )
        residual, analysis_frames, extension_frames = diagnostic._extract_real_residual(
            waveform,
            cepstrum,
            expected_samples=samples,
        )
        self.assertEqual(analysis_frames, frames + 1)
        self.assertEqual(extension_frames, 1)
        self.assertEqual(residual.shape, waveform.shape)
        self.assertLess(float((residual - waveform).abs().max()), 2.0e-5)

        resynth = production.render_time_varying_minimum_phase(
            residual.unsqueeze(0),
            cepstrum.unsqueeze(0),
        ).squeeze(0)
        self.assertEqual(resynth.shape, waveform.shape)
        self.assertLess(float((resynth - waveform).abs().max()), 3.0e-5)

    def test_scope_is_three_heldout_items_and_order_64(self) -> None:
        source = inspect.getsource(diagnostic.run_real_residual_resynthesis)
        self.assertEqual(diagnostic.DEFAULT_SPLIT, "val")
        self.assertEqual(diagnostic.DEFAULT_ITEMS, 3)
        self.assertEqual(diagnostic.CEPSTRAL_ORDER, 64)
        self.assertEqual(diagnostic.N_FFT, 1024)
        self.assertEqual(diagnostic.HOP_LENGTH, 256)
        self.assertEqual(diagnostic.SAMPLE_RATE, 24000)
        self.assertIn("collect_owned_vocoder_utterances", inspect.getsource(diagnostic))
        self.assertIn("reference_log_magnitude_to_one_sided_cepstrum", source)
        self.assertIn("one_sided_real_cepstrum_to_minimum_phase_fir", source)
        self.assertIn("render_time_varying_minimum_phase", source)
        self.assertIn("__real_residual_resynthesis.wav", source)
        self.assertIn("vocoder_minimum_phase_oracle_real_residual_v1", source)

    def test_only_real_residual_is_used_as_resynthesis_excitation(self) -> None:
        source = inspect.getsource(diagnostic.run_real_residual_resynthesis)
        full_source = inspect.getsource(diagnostic)
        self.assertIn("residual.unsqueeze(0)", source)
        self.assertNotIn("render_owned_minimum_phase_vocoder_path(", full_source)
        self.assertNotIn("build_neutral_excitation(", full_source)
        self.assertNotIn("_deterministic_aperiodic_noise(", full_source)
        self.assertNotIn("_gaussian_aperiodic_noise(", full_source)
        self.assertIn('"synthetic_excitation_used": False', full_source)
        self.assertIn('"build_neutral_excitation_used": False', full_source)

    def test_report_records_terminal_stft_alignment_and_policy_state(self) -> None:
        source = inspect.getsource(diagnostic)
        self.assertIn(
            '"terminal_transfer_extension_rule": "repeat_last_conditioning_transfer_only_for_centered_stft_terminal_frames"',
            source,
        )
        self.assertIn('"production_renderer_modified": False', source)
        self.assertIn('"model_used": False', source)
        self.assertIn('"training_executed": False', source)
        self.assertIn('"optimizer_created": False', source)
        self.assertIn('"checkpoint_loaded": False', source)
        self.assertIn('"checkpoint_written": False', source)
        self.assertIn('"predicted_duration_modified": False', source)
        self.assertIn('"posthoc_gain_normalization_used": False', source)
        self.assertIn('"posthoc_eq_used": False', source)
        self.assertIn('"posthoc_denoising_used": False', source)

    def test_diagnostic_has_no_training_external_model_or_postprocess_path(self) -> None:
        lowered = inspect.getsource(diagnostic).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "torch.load(",
            "from_pretrained",
            "vocos",
            "hifigan",
            "bigvgan",
            "cuda",
            "normalize(",
            "equalizer",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
