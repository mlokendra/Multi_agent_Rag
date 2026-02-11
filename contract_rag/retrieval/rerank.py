from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from contract_rag.retrieval.retriever import RetrievedChunk


def rerank(query: str, results: "List[RetrievedChunk]") -> "List[RetrievedChunk]":
    """Minimal reranker: prefer higher keyword/BM25, then fused score."""
    return sorted(results, key=lambda r: (round(r.keyword_score, 3), r.score), reverse=True)
