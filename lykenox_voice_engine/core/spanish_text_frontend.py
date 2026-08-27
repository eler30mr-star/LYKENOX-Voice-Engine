"""Versioned Spanish phoneme frontend owned by LYKENOX.

The public contract stays stable while pronunciation rules live in the local
``spanish_g2p`` module. Training and runtime code consume phoneme IDs directly;
no external TTS frontend is required.
"""

from __future__ import annotations

from dataclasses import dataclass

from lykenox_voice_engine.core.spanish_g2p import (
    FRONTEND_VERSION,
    TOKEN_TO_ID,
    VOCAB,
    normalize_spanish_text,
    text_to_phonemes,
)


@dataclass(frozen=True)
class FrontendOutput:
    """Normalized text plus the exact phoneme tokens consumed by LYKENOX Speech."""

    normalized_text: str
    tokens: list[str]
    token_ids: list[int]
    frontend_version: str = FRONTEND_VERSION


class SpanishTextFrontend:
    """Stable LYKENOX Spanish frontend interface.

    ``process`` exposes the full versioned phoneme result and ``encode`` returns
    token IDs for acoustic-model consumption. The implementation is local to
    LYKENOX and can evolve by version without changing the product API.
    """

    version = FRONTEND_VERSION

    def normalize(self, text: str) -> str:
        return normalize_spanish_text(text)

    def process(self, text: str) -> FrontendOutput:
        normalized, tokens = text_to_phonemes(text)
        return FrontendOutput(
            normalized_text=normalized,
            tokens=tokens,
            token_ids=[TOKEN_TO_ID.get(token, TOKEN_TO_ID["<unk>"]) for token in tokens],
        )

    def encode(self, text: str) -> list[int]:
        return self.process(text).token_ids

    def vocabulary(self) -> dict[str, int]:
        return dict(TOKEN_TO_ID)

    @property
    def vocab_size(self) -> int:
        return len(VOCAB)


def encode_spanish_text(text: str) -> FrontendOutput:
    """Functional compatibility wrapper around the product frontend contract."""

    return SpanishTextFrontend().process(text)


def vocabulary() -> dict[str, int]:
    """Functional compatibility wrapper for callers that only need the vocabulary."""

    return SpanishTextFrontend().vocabulary()
