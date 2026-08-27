"""Deterministic Spanish text frontend owned by LYKENOX.

This is the stable product-side normalization/token contract for the first speech
prototype. It deliberately does not call a third-party TTS frontend. A future
phoneme frontend can replace the tokenizer behind the same contract.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<space>"]
SPANISH_SYMBOLS = list("abcdefghijklmnñopqrstuvwxyzáéíóúü0123456789.,;:!?¿¡-'()")
VOCAB = SPECIAL + SPANISH_SYMBOLS
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


@dataclass(frozen=True)
class FrontendOutput:
    normalized_text: str
    tokens: list[str]
    token_ids: list[int]


def normalize_spanish_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def encode_spanish_text(text: str) -> FrontendOutput:
    normalized = normalize_spanish_text(text)
    tokens = ["<bos>"]
    for char in normalized:
        tokens.append("<space>" if char == " " else char if char in TOKEN_TO_ID else "<unk>")
    tokens.append("<eos>")
    return FrontendOutput(
        normalized_text=normalized,
        tokens=tokens,
        token_ids=[TOKEN_TO_ID[token] for token in tokens],
    )


def vocabulary() -> dict[str, int]:
    return dict(TOKEN_TO_ID)
