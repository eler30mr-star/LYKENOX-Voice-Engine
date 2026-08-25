"""Spanish Lite reclist helpers for the sample-based backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Reclist:
    """Loaded voicebank reclist with unique aliases in recording order."""

    name: str
    aliases: tuple[str, ...]

    def contains(self, alias: str) -> bool:
        """Return whether an alias is part of the reclist."""

        return alias.lower() in set(self.aliases)


def load_reclist(path: Path, name: str = "LYKENOX Spanish Lite") -> Reclist:
    """Load a UTF-8 reclist and remove comments, blanks, and duplicates."""

    seen: set[str] = set()
    aliases: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        alias = raw_line.strip().lower()
        if not alias or alias.startswith("#") or alias in seen:
            continue
        seen.add(alias)
        aliases.append(alias)
    return Reclist(name=name, aliases=tuple(aliases))


def missing_aliases(required: list[str], available: set[str]) -> list[str]:
    """Return sorted required aliases missing from a voicebank."""

    return sorted({alias.lower() for alias in required if alias.lower() not in available})
