from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from contract_rag.agents.retriever import EvidenceChunk


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    top_n_in: int = 30
    top_k_out: int = 10


class Reranker:
    """
    Optional reranker step.

    Default behavior: no-op (keeps original order).
    You can integrate a cross-encoder reranker later (bge-reranker, cohere rerank, etc.).
    """

    def __init__(self, config: Optional[RerankConfig] = None) -> None:
        self.config = config or RerankConfig(enabled=False)

    def rerank(self, query: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        if not self.config.enabled or not chunks:
            return chunks[: self.config.top_k_out]

        # Placeholder: If you plug a reranker, compute rerank_score and sort.
        # For now, we sort by existing score descending.
        ranked = sorted(chunks[: self.config.top_n_in], key=lambda c: c.score, reverse=True)
        return ranked[: self.config.top_k_out]
