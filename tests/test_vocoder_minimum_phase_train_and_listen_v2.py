from __future__ import annotations

import inspect
import unittest

from lykenox_voice_engine.training import speech_vocoder_minimum_phase_noise as noise
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_and_listen as pipeline
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_v3_decision as decision
from lykenox_voice_engine.training import speech_vocoder_minimum_phase_train_v2 as historical_v2


class MinimumPhaseTrainAndListenV2HistoricalTests(unittest.TestCase):
    def test_v2_is_recorded_as_failed_without_optimizer_or_checkpoint(self) -> None:
        self.assertEqual(decision.V2_PREFLIGHT_STATUS, "fail")
        self.assertFalse(decision.V2_PREFLIGHT_OPTIMIZER_CREATED)
        self.assertFalse(decision.V2_PREFLIGHT_PARAMETER_UPDATE_EXECUTED)
        self.assertFalse(decision.V2_PREFLIGHT_CHECKPOINT_SAVED)
        self.assertEqual(
            decision.V2_WEIGHT_CONTRACT_STATUS,
            "rejected_for_training_directional_conflict",
        )

    def test_failure_evidence_contains_negative_directions(self) -> None:
        evidence = decision.V2_FAILURE_EVIDENCE
        self.assertLess(evidence["cepstrum_neutral_spectral_balance_alignment"], 0.0)
        self.assertLess(evidence["cepstrum_connected_presence_descent_dot"], 0.0)
        self.assertLess(evidence["parameter_neutral_envelope_alignment"], 0.0)
        self.assertLess(evidence["parameter_connected_presence_descent_dot"], 0.0)

    def test_historical_v2_trainer_remains_for_forensics_but_is_not_pipeline_active(self) -> None:
        historical_source = inspect.getsource(historical_v2)
        self.assertIn("run_v2_authority_preflight", historical_source)
        pipeline_source = inspect.getsource(pipeline)
        self.assertNotIn("run_minimum_phase_training_v2", pipeline_source)
        self.assertIn("run_minimum_phase_training_v3", pipeline_source)
        self.assertEqual(
            pipeline.PIPELINE_VERSION,
            "owned-minimum-phase-train-and-listen-v3-directional-fixed",
        )

    def test_per_example_noise_seed_remains_deterministic(self) -> None:
        first = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0001", start_frame=12
        )
        repeat = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0001", start_frame=12
        )
        other = noise.stable_owned_noise_seed(
            97, split="train", utterance_id="speech_0002", start_frame=12
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
