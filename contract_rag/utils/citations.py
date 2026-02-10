from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Citation:
    chunk_id: str
    source: str
    text: str
    score: float


def format_citations(results: List[Citation]) -> str:
    unique: Dict[str, Citation] = {}
    for res in results:
        if res.chunk_id not in unique:
            unique[res.chunk_id] = res
    items = [f"[{i+1}] {c.source} (score={c.score:.2f})" for i, c in enumerate(unique.values())]
    return "\n".join(items)


def attach_citations(answer: str, results: List[Citation]) -> str:
    """Append citation block to an answer string."""
    if not results:
        return answer
    block = format_citations(results)
    return f"{answer}\n\nCitations:\n{block}"
