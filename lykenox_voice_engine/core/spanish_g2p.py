"""Deterministic Spanish grapheme-to-phoneme rules owned by LYKENOX.

This is a compact Latin-American Spanish bootstrap for the product frontend. It
has no dependency on an external TTS frontend. The rule set is intentionally
versioned and replaceable behind the stable ``SpanishTextFrontend`` contract.
"""

from __future__ import annotations

import re
import unicodedata

FRONTEND_VERSION = "es-phoneme-v1"

SHORT_PAUSE_PUNCT = {",", ";", ":", "-"}
LONG_PAUSE_PUNCT = {".", "!", "?"}
LEADING_PUNCT = {"¿", "¡"}

PHONEMES = [
    "a", "e", "i", "o", "u",
    "b", "k", "ch", "d", "f", "g", "x",
    "l", "y", "m", "n", "ny", "p", "r", "rr",
    "s", "t", "w",
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<wb>", "<pau_short>", "<pau_long>"]
VOCAB = SPECIAL_TOKENS + PHONEMES
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}

_VOWEL_MAP = {
    "a": "a", "á": "a",
    "e": "e", "é": "e",
    "i": "i", "í": "i",
    "o": "o", "ó": "o",
    "u": "u", "ú": "u",
}

_DIGIT_WORDS = {
    "0": "cero",
    "1": "uno",
    "2": "dos",
    "3": "tres",
    "4": "cuatro",
    "5": "cinco",
    "6": "seis",
    "7": "siete",
    "8": "ocho",
    "9": "nueve",
}


def normalize_spanish_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _expand_digit_run(run: str) -> list[str]:
    """Spell digits individually for deterministic bootstrap number handling."""

    words: list[str] = []
    for digit in run:
        words.append(_DIGIT_WORDS.get(digit, ""))
    return [word for word in words if word]


def _scan_units(text: str) -> list[tuple[str, str]]:
    """Split normalized text into words, punctuation, and numeric runs."""

    units: list[tuple[str, str]] = []
    for match in re.finditer(r"[a-záéíóúüñ]+|[0-9]+|[.,;:!?¿¡-]", text):
        value = match.group(0)
        if value[0].isdigit():
            units.append(("number", value))
        elif value in SHORT_PAUSE_PUNCT or value in LONG_PAUSE_PUNCT or value in LEADING_PUNCT:
            units.append(("punct", value))
        else:
            units.append(("word", value))
    return units


def word_to_phonemes(word: str) -> list[str]:
    """Convert one normalized Spanish word to a compact phoneme sequence.

    The pronunciation target is seseo/yeismo, suitable as the first LYKENOX
    Latin-American Spanish voice. This is deterministic and intentionally small;
    lexical exceptions can be added later without changing the public contract.
    """

    if not word:
        return []
    if word == "y":
        return ["i"]

    output: list[str] = []
    i = 0
    while i < len(word):
        char = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        nxt2 = word[i + 2] if i + 2 < len(word) else ""

        if char in _VOWEL_MAP:
            output.append(_VOWEL_MAP[char])
            i += 1
            continue
        if char == "ü":
            output.append("w")
            i += 1
            continue
        if char == "h":
            i += 1
            continue
        if char in {"b", "v"}:
            output.append("b")
            i += 1
            continue
        if char == "p":
            output.append("p")
            i += 1
            continue
        if char == "t":
            output.append("t")
            i += 1
            continue
        if char == "d":
            output.append("d")
            i += 1
            continue
        if char == "f":
            output.append("f")
            i += 1
            continue
        if char == "m":
            output.append("m")
            i += 1
            continue
        if char == "n":
            output.append("n")
            i += 1
            continue
        if char == "ñ":
            output.append("ny")
            i += 1
            continue
        if char == "l":
            if nxt == "l":
                output.append("y")
                i += 2
            else:
                output.append("l")
                i += 1
            continue
        if char == "c":
            if nxt == "h":
                output.append("ch")
                i += 2
            else:
                output.append("s" if nxt in {"e", "é", "i", "í"} else "k")
                i += 1
            continue
        if char == "q":
            output.append("k")
            if nxt == "u" and nxt2 in {"e", "é", "i", "í"}:
                i += 2
            else:
                i += 1
            continue
        if char == "g":
            if nxt == "ü" and nxt2 in {"e", "é", "i", "í"}:
                output.extend(("g", "w"))
                i += 2
            elif nxt == "u" and nxt2 in {"e", "é", "i", "í"}:
                output.append("g")
                i += 2
            else:
                output.append("x" if nxt in {"e", "é", "i", "í"} else "g")
                i += 1
            continue
        if char == "j":
            output.append("x")
            i += 1
            continue
        if char in {"s", "z"}:
            output.append("s")
            i += 1
            continue
        if char == "x":
            output.extend(("k", "s"))
            i += 1
            continue
        if char == "w":
            output.append("w")
            i += 1
            continue
        if char == "y":
            output.append("i" if i == len(word) - 1 else "y")
            i += 1
            continue
        if char == "r":
            if nxt == "r":
                output.append("rr")
                i += 2
            else:
                previous = word[i - 1] if i > 0 else ""
                strong = i == 0 or previous in {"n", "l", "s"}
                output.append("rr" if strong else "r")
                i += 1
            continue

        output.append("<unk>")
        i += 1

    return output


def text_to_phonemes(text: str) -> tuple[str, list[str]]:
    """Normalize Spanish text and return the full versioned phoneme token stream."""

    normalized = normalize_spanish_text(text)
    tokens: list[str] = ["<bos>"]
    previous_was_word = False

    for kind, value in _scan_units(normalized):
        if kind == "punct":
            if value in LEADING_PUNCT:
                continue
            pause = "<pau_short>" if value in SHORT_PAUSE_PUNCT else "<pau_long>"
            if len(tokens) > 1 and tokens[-1] not in {"<pau_short>", "<pau_long>", "<wb>"}:
                tokens.append(pause)
            previous_was_word = False
            continue

        words = _expand_digit_run(value) if kind == "number" else [value]
        for word in words:
            if previous_was_word and tokens[-1] != "<wb>":
                tokens.append("<wb>")
            tokens.extend(word_to_phonemes(word))
            previous_was_word = True

    if len(tokens) > 1 and tokens[-1] == "<wb>":
        tokens.pop()
    tokens.append("<eos>")
    return normalized, tokens
