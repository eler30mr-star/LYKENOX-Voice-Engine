"""Deterministic Spanish text frontend owned by LYKENOX.

This module is the stable product-side normalization/token contract for the first
speech model. It deliberately does not call a third-party TTS frontend. A future
phoneme/G2P implementation can replace the internals behind ``SpanishTextFrontend``
without changing the dataset, training, runtime, or API contracts.
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
    """Normalized text plus the exact tokens consumed by LYKENOX Speech."""

    normalized_text: str
    tokens: list[str]
    token_ids: list[int]


class SpanishTextFrontend:
    """Stable LYKENOX Spanish frontend interface.

    Training and inference code should depend on this class instead of on a
    trainer-specific tokenizer. ``process`` exposes the complete frontend result,
    while ``encode`` returns token IDs for model consumption.
    """

    def normalize(self, text: str) -> str:
        return normalize_spanish_text(text)

    def process(self, text: str) -> FrontendOutput:
        normalized = self.normalize(text)
        tokens = ["<bos>"]
        for char in normalized:
            if char == " ":
                tokens.append("<space>")
            elif char in TOKEN_TO_ID:
                tokens.append(char)
            else:
                tokens.append("<unk>")
        tokens.append("<eos>")
        return FrontendOutput(
            normalized_text=normalized,
            tokens=tokens,
            token_ids=[TOKEN_TO_ID[token] for token in tokens],
        )

    def encode(self, text: str) -> list[int]:
        return self.process(text).token_ids

    def vocabulary(self) -> dict[str, int]:
        return dict(TOKEN_TO_ID)

    @property
    def vocab_size(self) -> int:
        return len(TOKEN_TO_ID)


def normalize_spanish_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def encode_spanish_text(text: str) -> FrontendOutput:
    """Functional compatibility wrapper around the product frontend contract."""

    return SpanishTextFrontend().process(text)


def vocabulary() -> dict[str, int]:
    """Functional compatibility wrapper for callers that only need the vocabulary."""

    return SpanishTextFrontend().vocabulary()
