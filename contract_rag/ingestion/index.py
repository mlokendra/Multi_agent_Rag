from __future__ import annotations

import logging
import math
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np

from contract_rag.ingestion.chunk import Chunk
from contract_rag.utils.textnorm import normalize_text

logger = logging.getLogger(__name__)


class SimpleVectorizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return normalize_text(text).lower().split()

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        docs_tokens = [self._tokenize(t) for t in texts]
        # Build vocab
        for tokens in docs_tokens:
            for tok in tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        if not self.vocab:
            return np.zeros((len(texts), 0), dtype=np.float32)

        # Term frequency matrix
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        df = np.zeros(len(self.vocab), dtype=np.float32)
        for i, tokens in enumerate(docs_tokens):
            counts = Counter(tokens)
            for tok, freq in counts.items():
                j = self.vocab[tok]
                mat[i, j] = freq
                df[j] += 1
        # IDF
        n_docs = len(texts)
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1
        mat *= self.idf
        mat = self._l2_normalize(mat)
        return mat

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        docs_tokens = [self._tokenize(t) for t in texts]
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, tokens in enumerate(docs_tokens):
            counts = Counter(tokens)
            for tok, freq in counts.items():
                j = self.vocab.get(tok)
                if j is None:
                    continue
                mat[i, j] = freq
        if self.idf is not None and self.idf.size:
            mat *= self.idf
        return self._l2_normalize(mat)

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms


class VectorIndex:
    def __init__(self) -> None:
        self.vectorizer = SimpleVectorizer()
        self.embeddings: np.ndarray | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.embeddings = self.vectorizer.fit_transform(texts)
        logger.info("Built index with %d chunks", len(self.chunks))

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        q_emb = self.vectorizer.transform([query])
        scores = (self.embeddings @ q_emb.T).ravel()
        top_idx = np.argsort(scores)[::-1][:k]
        results: List[Tuple[Chunk, float]] = []
        for idx in top_idx:
            results.append((self.chunks[int(idx)], float(scores[int(idx)])))
        return results
