from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from contract_rag.config import settings
from contract_rag.ingestion.chunk import Chunk, chunk_all
from contract_rag.ingestion.index import VectorIndex
from contract_rag.ingestion.load import load_documents
from contract_rag.retrieval import rerank
from contract_rag.utils.textnorm import normalize_text

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "that", "this", "those", "these",
    "as", "at", "from", "it", "its", "into", "about", "than", "then", "so", "if",
}


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    keyword_score: float
    vector_score: float
    bm25_score: float


class HybridRetriever:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.index = VectorIndex()
        self.chunks: List[Chunk] = []
        self.chunk_tokens: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_dl: float = 0.0
        self._id_to_chunk: Dict[str, Chunk] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._prepared = False

    def prepare(self) -> None:
        docs = load_documents(self.data_dir)
        self.chunks = chunk_all(docs, settings.target_chunk_size, settings.chunk_overlap)
        self.index.build(self.chunks)
        self._id_to_chunk = {c.id: c for c in self.chunks}
        self._id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}
        self._build_bm25_cache()
        self._prepared = True
        logger.info("Prepared retriever with %d documents -> %d chunks", len(docs), len(self.chunks))

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphanumeric tokenization; strips punctuation for better BM25 recall."""
        cleaned = normalize_text(text).lower()
        tokens = re.findall(r"[a-z0-9]+", cleaned)
        return [t for t in tokens if t and t not in STOPWORDS]

    def _build_bm25_cache(self) -> None:
        """Pre-compute token stats for simple BM25."""
        self.chunk_tokens = [self._tokenize(c.text) for c in self.chunks]
        doc_freq: Dict[str, int] = {}
        lengths: List[int] = []
        for tokens in self.chunk_tokens:
            lengths.append(len(tokens))
            for tok in set(tokens):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        self.doc_freq = doc_freq
        self.avg_dl = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _bm25_score(self, query_tokens: List[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25 for a single chunk index."""
        tokens = self.chunk_tokens[idx]
        if not tokens:
            return 0.0
        freq: Dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
        score = 0.0
        dl = len(tokens)
        for tok in query_tokens:
            df = self.doc_freq.get(tok)
            if not df:
                continue
            tf = freq.get(tok, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * (dl / (self.avg_dl + 1e-9)))
            score += idf * ((tf * (k1 + 1)) / denom)
        return score

    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedChunk]:
        if not self._prepared:
            self.prepare()
        top_k = k or settings.top_k
        vector_hits = self.index.search(query, top_k * 2)
        q_tokens = self._tokenize(query)

        # BM25 scores for all chunks (small corpora) then take top
        bm_scores = [self._bm25_score(q_tokens, idx) for idx in range(len(self.chunks))]
        bm_ranked_idx = sorted(range(len(self.chunks)), key=lambda i: bm_scores[i], reverse=True)[: top_k * 2]

        # Reciprocal rank fusion (RRF) between vector and BM25
        rrf_k = 60
        fused: Dict[str, float] = {}
        vector_score_map: Dict[str, float] = {}
        for rank, (chunk, v_score) in enumerate(vector_hits, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            vector_score_map[chunk.id] = v_score

        for rank, idx in enumerate(bm_ranked_idx, start=1):
            chunk = self.chunks[idx]
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (rrf_k + rank)

        candidate_ids = list(fused.keys())
        results: List[RetrievedChunk] = []
        for cid in candidate_ids:
            chunk = self._id_to_chunk[cid]
            v_score = vector_score_map.get(cid, 0.0)
            bm_score = bm_scores[self._id_to_idx[cid]]
            combined = fused[cid]
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=combined,
                    keyword_score=bm_score,
                    vector_score=v_score,
                    bm25_score=bm_score,
                )
            )

        results = sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
        results = rerank.rerank(query, results)
        return results
