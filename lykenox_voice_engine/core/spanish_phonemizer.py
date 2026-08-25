"""Practical Spanish phonemizer for LYKENOX Spanish Lite aliases."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_VOWELS = set("aeiou")
_DIPHTHONGS = {"ai", "au", "ei", "eu", "oi", "ou", "ia", "ie", "io", "ua", "ue", "ui", "uo"}
_ONSETS = ("ch", "rr", "ll")


@dataclass(frozen=True)
class PhonemizedText:
    """Spanish text converted to practical voicebank aliases."""

    words: tuple[str, ...]
    syllables: tuple[str, ...]
    aliases: tuple[str, ...]


class SpanishPhonemizer:
    """Convert Spanish lyrics into Lite voicebank aliases."""

    def phonemize(self, text: str) -> PhonemizedText:
        """Return words, syllables, and aliases for Spanish lyrics."""

        words = tuple(_tokenize(text))
        syllables: list[str] = []
        for word in words:
            syllables.extend(_syllabify(_normalize_word(word)))
        aliases = tuple(_alias_for_syllable(syllable) for syllable in syllables if syllable)
        return PhonemizedText(words=words, syllables=tuple(syllables), aliases=aliases)


def _tokenize(text: str) -> list[str]:
    """Extract Spanish words from a lyric line."""

    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text.lower())


def _normalize_word(word: str) -> str:
    """Normalize Spanish orthography into renderer-friendly graphemes."""

    normalized = unicodedata.normalize("NFC", word.lower().replace("ü", "u"))
    normalized = normalized.replace("ch", "CH")
    normalized = normalized.replace("gue", "gE").replace("gui", "gI")
    normalized = normalized.replace("h", "")
    normalized = normalized.replace("CH", "ch")
    normalized = normalized.replace("v", "b")
    normalized = normalized.replace("ll", "y")
    normalized = normalized.replace("qu", "k")
    normalized = re.sub(r"c(?=[eéií])", "s", normalized)
    normalized = re.sub(r"g(?=[eéií])", "j", normalized)
    normalized = normalized.replace("gE", "ge").replace("gI", "gi")
    normalized = normalized.replace("x", "ks")
    normalized = normalized.replace("á", "a").replace("é", "e").replace("í", "i")
    normalized = normalized.replace("ó", "o").replace("ú", "u")
    return normalized


def _syllabify(word: str) -> list[str]:
    """Split a normalized Spanish word into practical syllable aliases."""

    if not word:
        return []
    nuclei = _find_nuclei(word)
    if not nuclei:
        return [word]
    starts = [0]
    for previous, current in zip(nuclei, nuclei[1:]):
        gap_start = previous[1]
        gap_end = current[0]
        consonants = word[gap_start:gap_end]
        split_at = gap_end - 1 if consonants else current[0]
        if consonants in {"ch", "ll", "rr"}:
            split_at = gap_start
        if len(consonants) >= 2 and consonants[-2:] in {"br", "bl", "dr", "fr", "gr", "kr", "pl", "pr", "tr"}:
            split_at = gap_end - 2
        starts.append(max(previous[1], split_at))
    syllables: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(word)
        syllables.append(word[start:end])
    return [syllable for syllable in syllables if syllable]


def _find_nuclei(word: str) -> list[tuple[int, int]]:
    """Find vowel nuclei, keeping basic diphthongs together."""

    nuclei: list[tuple[int, int]] = []
    index = 0
    while index < len(word):
        if word[index] not in _VOWELS:
            index += 1
            continue
        end = index + 1
        if end < len(word) and word[index : end + 1] in _DIPHTHONGS:
            end += 1
        nuclei.append((index, end))
        index = end
    return nuclei


def _alias_for_syllable(syllable: str) -> str:
    """Map normalized syllables to Spanish Lite aliases."""

    syllable = syllable.lower()
    for onset in _ONSETS:
        if syllable.startswith(onset):
            return onset + syllable[len(onset) :]
    if syllable.startswith("r") and syllable != "r" and len(syllable) > 1:
        return syllable
    return syllable



