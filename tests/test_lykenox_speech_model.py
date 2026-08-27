from __future__ import annotations

import unittest

import torch

from lykenox_voice_engine.core.spanish_text_frontend import (
    SpanishTextFrontend,
    encode_spanish_text,
    vocabulary,
)
from lykenox_voice_engine.models.speech import LykenoxSpeechAcousticModel, LykenoxSpeechConfig


class LykenoxSpeechModelTests(unittest.TestCase):
    def test_spanish_frontend_is_deterministic(self) -> None:
        encoded = encode_spanish_text("¡Hola, LYKENOX!")
        self.assertEqual(encoded.normalized_text, "¡hola, lykenox!")
        self.assertEqual(encoded.frontend_version, "es-phoneme-v1")
        self.assertEqual(encoded.tokens[0], "<bos>")
        self.assertEqual(encoded.tokens[-1], "<eos>")
        self.assertEqual(len(encoded.tokens), len(encoded.token_ids))
        self.assertGreater(len(vocabulary()), 20)
        self.assertNotIn("<unk>", encoded.tokens)

    def test_spanish_frontend_class_matches_functional_contract(self) -> None:
        frontend = SpanishTextFrontend()
        processed = frontend.process("  Hola   mundo  ")
        self.assertEqual(processed.normalized_text, "hola mundo")
        self.assertEqual(frontend.encode("  Hola   mundo  "), processed.token_ids)
        self.assertEqual(frontend.vocabulary(), vocabulary())
        self.assertEqual(frontend.vocab_size, len(vocabulary()))

    def test_model_forward_shapes(self) -> None:
        config = LykenoxSpeechConfig(vocab_size=128, hidden_size=64, encoder_layers=2, encoder_heads=4)
        model = LykenoxSpeechAcousticModel(config)
        token_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
        mask = torch.ones_like(token_ids, dtype=torch.bool)
        durations = torch.tensor([[2, 3, 2, 1]], dtype=torch.long)
        output = model(token_ids, mask, durations)
        self.assertEqual(tuple(output["mel"].shape), (1, 8, config.mel_bins))
        self.assertEqual(tuple(output["duration_prediction"].shape), (1, 4))
        self.assertGreater(model.parameter_count(), 0)

    def test_teacher_durations_are_not_clipped_by_inference_limit(self) -> None:
        config = LykenoxSpeechConfig(
            vocab_size=128,
            hidden_size=32,
            encoder_layers=1,
            encoder_heads=4,
            max_duration_frames=5,
        )
        model = LykenoxSpeechAcousticModel(config)
        token_ids = torch.tensor([[5, 6]], dtype=torch.long)
        durations = torch.tensor([[8, 3]], dtype=torch.long)
        output = model(token_ids, durations=durations)
        self.assertEqual(tuple(output["mel"].shape), (1, 11, config.mel_bins))

    def test_negative_teacher_duration_is_rejected(self) -> None:
        config = LykenoxSpeechConfig(vocab_size=128, hidden_size=32, encoder_layers=1, encoder_heads=4)
        model = LykenoxSpeechAcousticModel(config)
        token_ids = torch.tensor([[5, 6]], dtype=torch.long)
        durations = torch.tensor([[2, -1]], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(token_ids, durations=durations)

    def test_model_can_backpropagate(self) -> None:
        config = LykenoxSpeechConfig(vocab_size=128, hidden_size=32, encoder_layers=1, encoder_heads=4)
        model = LykenoxSpeechAcousticModel(config)
        token_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
        durations = torch.tensor([[2, 2, 2]], dtype=torch.long)
        output = model(token_ids, durations=durations)
        loss = output["mel"].abs().mean() + output["duration_prediction"].mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
