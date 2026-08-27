from __future__ import annotations

import unittest

from lykenox_voice_engine.core.spanish_g2p import text_to_phonemes, word_to_phonemes
from lykenox_voice_engine.core.spanish_text_frontend import SpanishTextFrontend


class SpanishG2PTests(unittest.TestCase):
    def test_core_spanish_rules(self) -> None:
        self.assertEqual(word_to_phonemes("queso"), ["k", "e", "s", "o"])
        self.assertEqual(word_to_phonemes("guitarra"), ["g", "i", "t", "a", "rr", "a"])
        self.assertEqual(word_to_phonemes("gente"), ["x", "e", "n", "t", "e"])
        self.assertEqual(word_to_phonemes("niño"), ["n", "i", "ny", "o"])
        self.assertEqual(word_to_phonemes("llave"), ["y", "a", "b", "e"])
        self.assertEqual(word_to_phonemes("perro"), ["p", "e", "rr", "o"])
        self.assertEqual(word_to_phonemes("rosa"), ["rr", "o", "s", "a"])
        self.assertEqual(word_to_phonemes("vergüenza"), ["b", "e", "r", "g", "w", "e", "n", "s", "a"])

    def test_text_has_boundaries_and_pauses(self) -> None:
        normalized, tokens = text_to_phonemes("Hola, mundo.")
        self.assertEqual(normalized, "hola, mundo.")
        self.assertEqual(tokens[0], "<bos>")
        self.assertEqual(tokens[-1], "<eos>")
        self.assertIn("<pau_short>", tokens)
        self.assertIn("<pau_long>", tokens)
        self.assertNotIn("<unk>", tokens)

    def test_frontend_is_versioned_and_deterministic(self) -> None:
        frontend = SpanishTextFrontend()
        first = frontend.process("¡Hola, LYKENOX!")
        second = frontend.process("¡Hola, LYKENOX!")
        self.assertEqual(first, second)
        self.assertEqual(first.frontend_version, "es-phoneme-v1")
        self.assertEqual(len(first.tokens), len(first.token_ids))
        self.assertLess(frontend.vocab_size, 128)


if __name__ == "__main__":
    unittest.main()
