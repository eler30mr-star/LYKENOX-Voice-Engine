"""Tests for Spanish phonemizer aliases."""

from __future__ import annotations

import unittest

from lykenox_voice_engine.core.spanish_phonemizer import SpanishPhonemizer


class TestSpanishPhonemizer(unittest.TestCase):
    """Validate practical Spanish alias conversion."""

    def test_baila_conmigo_aliases(self) -> None:
        result = SpanishPhonemizer().phonemize("baila conmigo")
        self.assertEqual(result.aliases, ("bai", "la", "con", "mi", "go"))

    def test_spanish_rules_cover_common_graphemes(self) -> None:
        result = SpanishPhonemizer().phonemize("Vaca queso guitarra hacha lluvia niño carro xilo")
        self.assertIn("ba", result.aliases)
        self.assertIn("ke", result.aliases)
        self.assertIn("gi", result.aliases)
        self.assertIn("cha", result.aliases)
        self.assertIn("ño", result.aliases)
        self.assertIn("rro", result.aliases)


if __name__ == "__main__":
    unittest.main()

