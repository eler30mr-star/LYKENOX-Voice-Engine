from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

from lykenox_voice_engine.training import speech_residual_codebook_v1 as codebook


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnostic_residual_codebook_oracle_v1.py"
SPEC = importlib.util.spec_from_file_location("diagnostic_residual_codebook_oracle_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import diagnostic script: {SCRIPT_PATH}")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


class ResidualCodebookOracleTests(unittest.TestCase):
    def test_analysis_synthesis_identity(self) -> None:
        samples = 16 * codebook.HOP_LENGTH
        index = torch.arange(samples, dtype=torch.float32)
        residual = 0.2 * torch.sin(index * 0.071) + 0.07 * torch.sin(index * 0.193)
        vectors = codebook.residual_analysis_vectors(residual)
        reconstructed = codebook.residual_synthesis_from_analysis_vectors(
            vectors,
            output_samples=samples,
        )
        self.assertEqual(reconstructed.shape, residual.shape)
        self.assertLess(float((reconstructed - residual).abs().max()), 2.0e-5)

    def test_codebook_builder_is_train_only_owned_and_not_training(self) -> None:
        source = inspect.getsource(codebook.build_owned_residual_codebook)
        full = inspect.getsource(codebook)
        self.assertIn('if split != "train"', source)
        self.assertIn('"source_split": split', source)
        self.assertIn('"third_party_voice_data_used": False', source)
        self.assertIn('"third_party_model_or_checkpoint_used": False', source)
        self.assertIn('"training_executed": False', source)
        self.assertIn('"optimizer_created": False', source)
        lowered = full.lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "from_pretrained",
            "vocos",
            "hifigan",
            "bigvgan",
            "cuda",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_oracle_is_heldout_analysis_by_synthesis_only(self) -> None:
        source = inspect.getsource(oracle.run_residual_codebook_oracle)
        full = inspect.getsource(oracle)
        self.assertIn('if split == "train"', source)
        self.assertIn('metadata.get("source_split") != "train"', source)
        self.assertIn('"heldout_residual_used_only_as_oracle_search_target": True', source)
        self.assertIn('"heldout_residual_added_to_codebook": False', source)
        self.assertIn('"oracle_indices_or_gains_valid_for_product_inference": False', source)
        self.assertIn('"analysis_by_synthesis_oracle_only": True', source)
        self.assertIn('"human_full_utterance_listening_required": True', source)
        self.assertIn('render_time_varying_minimum_phase', source)
        self.assertNotIn('build_neutral_excitation(', full)

    def test_oracle_gain_is_bounded_nonnegative(self) -> None:
        target = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        words = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        indices = torch.tensor([0, 1], dtype=torch.long)
        selected, selected_index, gain, mse = oracle._oracle_select_codevector(target, words, indices)
        self.assertEqual(selected_index, 0)
        self.assertGreaterEqual(gain, 0.0)
        self.assertLessEqual(gain, oracle.MAX_ORACLE_GAIN)
        self.assertLess(mse, 1.0e-7)
        self.assertTrue(torch.allclose(selected, target))

    def test_cpu_only_no_external_or_posthoc_paths(self) -> None:
        lowered = (inspect.getsource(codebook) + inspect.getsource(oracle)).lower()
        for forbidden in (
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "from_pretrained",
            "vocos",
            "hifigan",
            "bigvgan",
            "cuda",
            "equalizer",
            "denoise(",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
