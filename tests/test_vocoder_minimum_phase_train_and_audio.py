from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from lykenox_voice_engine.models.vocoder.network_minimum_phase_v1 import (
    LykenoxFrameRateCepstralPredictorV1,
)
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_artifact as artifact
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_heldout_audio as heldout
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train as trainer
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_and_listen as pipeline
from lykenox_voice_engine.training.speech_vocoder_minimum_phase_objective import (
    ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
)


class MinimumPhaseTrainAndAudioTests(unittest.TestCase):
    def test_trainer_is_wired_only_to_minimum_phase_v2_objective(self) -> None:
        source = inspect.getsource(trainer)
        lowered = source.lower()
        self.assertIn("OwnedMinimumPhaseObjectiveV2", source)
        self.assertIn("ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION", source)
        self.assertIn("render_owned_minimum_phase_vocoder_path", source)
        self.assertNotIn("speech_vocoder_loss_v2_weight_contract import", source)
        self.assertNotIn("combine_owned_vocoder_loss_v2", source)
        self.assertNotIn("FROZEN_WEIGHTS", source)
        self.assertNotIn("from_pretrained", lowered)
        self.assertNotIn("vocos", lowered)
        self.assertNotIn("bigvgan", lowered)
        self.assertNotIn("hifigan", lowered)

    def test_trainer_has_bounded_update_budget_and_deterministic_order(self) -> None:
        self.assertEqual(trainer.DEFAULT_MAX_UPDATES, 400)
        first = trainer._epoch_order(12, order_seed=17, epoch=3)
        second = trainer._epoch_order(12, order_seed=17, epoch=3)
        different = trainer._epoch_order(12, order_seed=17, epoch=4)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(sorted(first), list(range(12)))

    def test_checkpoint_roundtrip_preserves_exact_optimizer_resume(self) -> None:
        torch.manual_seed(1234)
        mel = torch.randn(1, 16, 80)
        f0 = torch.full((1, 16), 140.0)
        voiced = torch.ones(1, 16)
        periodicity = torch.full((1, 16), 0.8)

        def make_model_optimizer() -> tuple[LykenoxFrameRateCepstralPredictorV1, torch.optim.Optimizer]:
            torch.manual_seed(99)
            model = LykenoxFrameRateCepstralPredictorV1().cpu()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            return model, optimizer

        def step(model: LykenoxFrameRateCepstralPredictorV1, optimizer: torch.optim.Optimizer) -> None:
            optimizer.zero_grad(set_to_none=True)
            cepstrum = model(mel, f0, voiced, periodicity)
            loss = (cepstrum - 0.1).square().mean()
            loss.backward()
            optimizer.step()

        uninterrupted_model, uninterrupted_optimizer = make_model_optimizer()
        for _ in range(3):
            step(uninterrupted_model, uninterrupted_optimizer)

        resumed_model, resumed_optimizer = make_model_optimizer()
        step(resumed_model, resumed_optimizer)
        run_config = {"contract": ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION, "case": "unit-resume"}
        progress = {
            "epoch": 1,
            "next_item_offset": 2,
            "global_step": 1,
            "best_val_total": 1.0,
            "best_step": 0,
            "initial_validation": {"total": 1.0},
            "clipped_update_count": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            artifact.save_minimum_phase_checkpoint(
                path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                run_config=run_config,
                progress=progress,
                history=[{"step": 1}],
            )
            loaded_model, payload = artifact.load_minimum_phase_checkpoint(
                path,
                expected_run_config=run_config,
            )
            loaded_optimizer = torch.optim.AdamW(
                loaded_model.parameters(), lr=1e-4, weight_decay=1e-5
            )
            loaded_optimizer.load_state_dict(payload["optimizer_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            for _ in range(2):
                step(loaded_model, loaded_optimizer)

        for expected, actual in zip(
            uninterrupted_model.parameters(), loaded_model.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_contract_binds_active_v2_weight_contract(self) -> None:
        contract = artifact.checkpoint_contract()
        self.assertEqual(
            contract["loss_weight_contract_version"],
            ACTIVE_LOSS_WEIGHT_CONTRACT_VERSION,
        )
        self.assertIn("minimum-phase", contract["renderer_version"])

    def test_heldout_audio_is_complete_validation_audio_without_posthoc_processing(self) -> None:
        source = inspect.getsource(heldout).lower()
        self.assertIn('split: str = "val"', inspect.getsource(heldout))
        self.assertIn('subtype="float"', source)
        self.assertIn('"metrics_accept_voice_quality": false', source)
        self.assertIn('"product_acceptance_requires_human_listening": true', source)
        self.assertIn("collect_owned_vocoder_utterances", source)
        self.assertIn("expected_samples = utterance.mel_frames * hop_length", source)
        self.assertNotIn("normalize(", source)
        self.assertNotIn("equalizer", source)
        self.assertNotIn("denoise", source.replace("posthoc_denoising_used", ""))
        self.assertNotIn("from_pretrained", source)

    def test_one_shot_pipeline_trains_then_renders_complete_val_audio(self) -> None:
        source = inspect.getsource(pipeline)
        lowered = source.lower()
        self.assertIn("run_minimum_phase_training", source)
        self.assertIn("render_heldout_audio", source)
        self.assertIn('split="val"', source)
        self.assertIn('"ready_for_listening"', source)
        self.assertIn('"metrics_accept_voice_quality": False', source)
        self.assertNotIn("speech_vocoder_loss_v2_weight_contract", lowered)
        self.assertNotIn("from_pretrained", lowered)


if __name__ == "__main__":
    unittest.main()
