from __future__ import annotations

from typing import List

#from contract_rag.retrieval.retriever import RetrievedChunk


def rerank(query: str, results: List[RetrievedChunk]) -> List[RetrievedChunk]:
    # Minimal reranker: prefer chunks with higher keyword overlap, then combined score.
    return sorted(results, key=lambda r: (round(r.keyword_score, 3), r.score), reverse=True)
