from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from contract_rag.agents.retriever import EvidenceChunk


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    top_n_in: int = 30
    top_k_out: int = 10
    rrf_k: int = 60  # larger = smoother fusion


class Reranker:
    """
    Optional reranker step.

    Default behavior: RRF fusion between vector and BM25 scores carried in EvidenceChunk.metadata.
    If scores are missing, it falls back to input order.
    You can still swap to a cross-encoder reranker later.
    """

    def __init__(self, config: Optional[RerankConfig] = None) -> None:
        self.config = config or RerankConfig()

    def rerank(self, query: str, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        if not self.config.enabled or not chunks:
            return chunks[: self.config.top_k_out]

        # Limit the pool
        pool = chunks[: self.config.top_n_in]

        # Build ranks for vector and bm25 scores (higher is better).
        def build_rank_map(key: str) -> Dict[str, int]:
            scored = []
            for c in pool:
                meta = c.metadata or {}
                val = meta.get(key)
                if val is None:
                    continue
                scored.append((c.chunk_id, float(val)))
            scored = sorted(scored, key=lambda t: t[1], reverse=True)
            return {cid: rank + 1 for rank, (cid, _) in enumerate(scored)}

        vec_rank = build_rank_map("vector_score")
        bm_rank = build_rank_map("bm25_score")
        # Build document-level ranks: aggregate per-doc scores (use max combined chunk score)
        doc_scores: Dict[str, float] = {}
        for c in pool:
            meta = c.metadata or {}
            v = float(meta.get("vector_score") or 0.0)
            b = float(meta.get("bm25_score") or 0.0)
            combined = v + b
            # keep the max combined score per document
            doc_scores[c.doc_id] = max(doc_scores.get(c.doc_id, 0.0), combined)
        # sort docs by score desc -> assign 1-based ranks
        sorted_docs = sorted(doc_scores.items(), key=lambda t: t[1], reverse=True)
        doc_rank: Dict[str, int] = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_docs)}

        fused_scores: Dict[str, float] = {}
        for c in pool:
            cid = c.chunk_id
            # Default large rank if missing to minimally influence fusion
            r_vec = vec_rank.get(cid, len(pool))
            r_bm = bm_rank.get(cid, len(pool))
            r_doc = doc_rank.get(c.doc_id, len(doc_rank) + 1)
            score = 0.0
            # Reciprocal Rank Fusion terms: vector, BM25, and document-level rank
            score += 1.0 / (self.config.rrf_k + r_vec)
            score += 1.0 / (self.config.rrf_k + r_bm)
            score += 1.0 / (self.config.rrf_k + r_doc)
            fused_scores[cid] = score

        ranked = sorted(pool, key=lambda c: fused_scores.get(c.chunk_id, 0.0), reverse=True)
        return ranked[: self.config.top_k_out]
