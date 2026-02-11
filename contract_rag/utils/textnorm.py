from __future__ import annotations

import re
from typing import Iterable

_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str, keep_newlines: bool = False) -> str:
    """Light text normalization to reduce noise before chunking/indexing.

    When ``keep_newlines`` is True, we collapse spaces but preserve paragraph
    breaks so section headings remain detectable (important for legal docs).
    """
    cleaned = text.replace("\uf0b7", " ")  # bullets from some exports
    cleaned = cleaned.replace("\u2022", " ")

    if keep_newlines:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

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
