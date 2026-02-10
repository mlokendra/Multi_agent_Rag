from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from contract_rag.config import settings
from contract_rag.ingestion.chunk import Chunk, chunk_all
from contract_rag.ingestion.index import VectorIndex
from contract_rag.ingestion.load import load_documents
from contract_rag.retrieval import rerank
from contract_rag.utils.textnorm import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    keyword_score: float
    vector_score: float


class HybridRetriever:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.index = VectorIndex()
        self.chunks: List[Chunk] = []
        self._prepared = False

    def prepare(self) -> None:
        docs = load_documents(self.data_dir)
        self.chunks = chunk_all(docs, settings.target_chunk_size, settings.chunk_overlap)
        self.index.build(self.chunks)
        self._prepared = True
        logger.info("Prepared retriever with %d documents -> %d chunks", len(docs), len(self.chunks))

    @staticmethod
    def _keyword_score(query: str, chunk: Chunk) -> float:
        q_tokens = set(normalize_text(query).lower().split())
        if not q_tokens:
            return 0.0
        text_tokens = set(chunk.text.lower().split())
        return len(q_tokens & text_tokens) / len(q_tokens)

    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedChunk]:
        if not self._prepared:
            self.prepare()
        top_k = k or settings.top_k
        vector_hits = self.index.search(query, top_k * 2)
        results: List[RetrievedChunk] = []
        for chunk, v_score in vector_hits:
            kw = self._keyword_score(query, chunk)
            combined = (settings.hybrid_weight * kw) + ((1 - settings.hybrid_weight) * v_score)
            results.append(RetrievedChunk(chunk=chunk, score=combined, keyword_score=kw, vector_score=v_score))
        # Sort and trim
        results = sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
        results = rerank.rerank(query, results)
        return results
