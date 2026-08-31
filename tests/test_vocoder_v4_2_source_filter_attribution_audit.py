from __future__ import annotations

import inspect
import unittest

import torch

from lykenox_voice_engine.models.vocoder import (
    LykenoxVocoderConfig,
    LykenoxVocoderGeneratorV42,
)
from lykenox_voice_engine.training.speech_vocoder_v4_2_source_filter_attribution_audit import (
    AUDIT_VERSION,
    VALIDATION_INDICES,
    VARIANTS,
    _forward_variant,
    run_v4_2_source_filter_attribution,
)


class V42SourceFilterAttributionAuditTests(unittest.TestCase):
    def test_contract_is_fixed_and_read_only(self) -> None:
        self.assertEqual(AUDIT_VERSION, "vocoder-v4-2-source-filter-attribution-v1")
        self.assertEqual(VALIDATION_INDICES, (0, 1, 2))
        self.assertEqual(
            VARIANTS,
            (
                "baseline",
                "no_harmonic",
                "no_aperiodic",
                "no_explicit_excitation",
                "mel_envelope_only",
            ),
        )
        source = inspect.getsource(run_v4_2_source_filter_attribution)
        self.assertIn("target_mel = batch.mel", source)
        self.assertIn("target_f0 = batch.f0_hz", source)
        self.assertIn("target_voiced = batch.voiced", source)
        self.assertIn("torch.equal(canonical, reproduced)", source)
        self.assertIn("checkpoints_unchanged", source)
        self.assertIn("mean_minus_reference_by_variant", source)
        self.assertIn("ablation_change_vs_baseline", source)
        self.assertIn(
            '"next_gate": "select_vocoder_replacement_architecture_from_source_filter_attribution"',
            source,
        )
        for forbidden in (
            "optimizer.step",
            ".backward(",
            "torch.save(",
            '"acoustic_training_authorized": True',
            '"vocoder_training_authorized": True',
            '"posthoc_gain_normalization_used": True',
            '"posthoc_eq_used": True',
            '"posthoc_denoising_used": True',
            '"predicted_duration_modified": True',
        ):
            self.assertNotIn(forbidden, source)

    def test_internal_baseline_reproduces_v4_2_bit_exact(self) -> None:
        torch.manual_seed(23)
        config = LykenoxVocoderConfig()
        model = LykenoxVocoderGeneratorV42(
            config,
            hidden_channels=32,
            conditioning_channels=32,
            harmonics=2,
            highpass_kernel_size=65,
        ).cpu().eval()
        mel = torch.randn(1, 4, config.mel_bins)
        f0 = torch.full((1, 4), 120.0)
        voiced = torch.ones((1, 4))
        with torch.inference_mode():
            canonical = model(mel, f0, voiced)
            reproduced, diagnostics = _forward_variant(model, mel, f0, voiced, "baseline")
            no_harmonic, _ = _forward_variant(model, mel, f0, voiced, "no_harmonic")
            no_aperiodic, _ = _forward_variant(model, mel, f0, voiced, "no_aperiodic")
            no_explicit, _ = _forward_variant(model, mel, f0, voiced, "no_explicit_excitation")
            envelope_only, _ = _forward_variant(model, mel, f0, voiced, "mel_envelope_only")
        self.assertTrue(torch.equal(canonical, reproduced))
        self.assertEqual(tuple(canonical.shape), (1, 4 * config.hop_length))
        for wave in (no_harmonic, no_aperiodic, no_explicit, envelope_only):
            self.assertEqual(tuple(wave.shape), tuple(canonical.shape))
            self.assertTrue(torch.isfinite(wave).all())
        self.assertGreaterEqual(float(diagnostics["source_gate_mean"]), 0.0)
        self.assertLessEqual(float(diagnostics["source_gate_mean"]), 1.0)

    def test_ablation_surface_targets_source_without_mutating_weights(self) -> None:
        source = inspect.getsource(_forward_variant)
        self.assertIn('variant in ("no_harmonic", "no_explicit_excitation", "mel_envelope_only")', source)
        self.assertIn("harmonic = torch.zeros_like(harmonic)", source)
        self.assertIn('variant in ("no_aperiodic", "no_explicit_excitation", "mel_envelope_only")', source)
        self.assertIn("noise = torch.zeros_like(noise)", source)
        self.assertIn('if variant == "mel_envelope_only":', source)
        self.assertIn("x = envelope_hidden", source)
        self.assertNotIn("load_state_dict", source)
        self.assertNotIn("state_dict()[", source)


if __name__ == "__main__":
    unittest.main()
