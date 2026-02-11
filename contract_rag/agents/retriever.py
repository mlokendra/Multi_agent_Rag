from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class EvidenceChunk:
    """
    A single retrieved chunk of text with metadata for citations.
    """
    chunk_id: str
    doc_id: str
    doc_label: str
    section_path: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] | None = None


class SearchBackend(Protocol):
    """
    Plug your real vector DB / hybrid search here.

    Must return a list of dicts with at least:
      - chunk_id, doc_id, doc_label, section_path, text, score
    """
    def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...


class Retriever:
    """
    Retrieves candidate evidence chunks for a query.

    This class is intentionally backend-agnostic.
    You can adapt it to Chroma/FAISS/BM25, etc.
    """

    def __init__(self, backend: SearchBackend | None = None, default_top_k: int = 10) -> None:
        self.backend = backend
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceChunk]:
        if self.backend is None:
            # Safe fallback so system doesn't crash while wiring backend.
            return []

        k = top_k or self.default_top_k
        rows = self.backend.search(query=query, top_k=k, filters=filters)

        chunks: List[EvidenceChunk] = []
        for r in rows:
            chunks.append(
                EvidenceChunk(
                    chunk_id=str(r.get("chunk_id", "")),
                    doc_id=str(r.get("doc_id", "")),
                    doc_label=str(r.get("doc_label", r.get("doc_id", ""))),
                    section_path=str(r.get("section_path", "")),
                    text=str(r.get("text", "")),
                    score=float(r.get("score", 0.0)),
                    metadata=dict(r.get("metadata", {}) or {}),
                )
            )
        return chunks
