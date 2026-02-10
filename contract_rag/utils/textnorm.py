from __future__ import annotations

import re
from typing import Iterable

_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Light text normalization to reduce noise before chunking/indexing."""
    cleaned = text.replace("\uf0b7", " ")  # bullets from some exports
    cleaned = cleaned.replace("\u2022", " ")
    cleaned = _whitespace_re.sub(" ", cleaned)
    return cleaned.strip()


def sentencize(text: str) -> list[str]:
    """Split into pseudo-sentences; keeps things simple for clause-aware chunking."""
    parts = re.split(r"(?<=[.;!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def flatten(iterable: Iterable[Iterable[str]]) -> list[str]:
    out: list[str] = []
    for part in iterable:
        out.extend(part)
    return out
