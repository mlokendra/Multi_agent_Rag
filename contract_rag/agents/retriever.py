from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from contract_rag.config import settings
from contract_rag.retrieval.retriever import HybridRetriever, RetrievedChunk


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


class Retriever:
    """
    Retrieves candidate evidence chunks for a query using the core HybridRetriever logic.
    """

    def __init__(self, hybrid: HybridRetriever | None = None, default_top_k: int | None = None) -> None:
        self.hybrid = hybrid or HybridRetriever()
        self.default_top_k = default_top_k or settings.top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceChunk]:
        k = top_k or self.default_top_k
        hits: List[RetrievedChunk] = self.hybrid.retrieve(query, k=k)

        chunks: List[EvidenceChunk] = []
        for h in hits:
            doc_label = h.chunk.source
            doc_id = doc_label.split(" §", 1)[0]
            chunks.append(
                EvidenceChunk(
                    chunk_id=h.chunk.id,
                    doc_id=doc_id,
                    doc_label=doc_label,
                    section_path=h.chunk.section_number or "",
                    text=h.chunk.text,
                    score=h.score,
                    metadata={"vector_score": h.vector_score, "bm25_score": h.bm25_score},
                )
            )
        return chunks
