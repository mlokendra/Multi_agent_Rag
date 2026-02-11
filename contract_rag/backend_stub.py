# Simple backend implementations for the agent-layer retriever.
from __future__ import annotations

from typing import List, Dict, Any

from contract_rag.retrieval.retriever import HybridRetriever


class DummyBackend:
    def search(self, query: str, top_k: int, filters=None):
        return []


class LocalHybridBackend:
    """Adapter that reuses the existing HybridRetriever for agent layer."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def search(self, query: str, top_k: int, filters=None) -> List[Dict[str, Any]]:
        hits = self.retriever.retrieve(query, k=top_k)
        rows: List[Dict[str, Any]] = []
        for h in hits:
            doc_label = h.chunk.source
            base_id = doc_label.split(" §", 1)[0]
            rows.append(
                {
                    "chunk_id": h.chunk.id,
                    "doc_id": base_id,
                    "doc_label": doc_label,
                    "section_path": h.chunk.section_number or "",
                    "text": h.chunk.text,
                    "score": h.score,
                }
            )
        return rows
