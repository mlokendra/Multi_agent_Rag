from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from contract_rag.ingestion.chunk import Chunk
from contract_rag.utils.textnorm import normalize_text

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class SentenceTransformerVectorizer:
    """
    Semantic embeddings using sentence-transformers all-MiniLM-L6-v2.
    
    Model: all-MiniLM-L6-v2
    - Dimension: 384
    - Parameters: 22M
    - Optimized for semantic similarity
    - Trained on 215M sentence pairs
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Loaded SentenceTransformer model: {model_name} ({EMBEDDING_DIM}D)")

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode texts into semantic embeddings.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray of shape (n_texts, 384) with normalized embeddings
        """
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        
        # Encode with batch processing
        embeddings = self.model.encode(
            texts_list,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization
            batch_size=32,
            show_progress_bar=False,
        )
        logger.info(f"Encoded {len(texts_list)} texts into {embeddings.shape} embeddings")
        return embeddings.astype(np.float32)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode texts (same as fit_transform; no fitting step needed).
        """
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        
        embeddings = self.model.encode(
            texts_list,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


class VectorIndex:
    """
    Vector index using semantic embeddings (all-MiniLM-L6-v2).
    
    Supports fast semantic similarity search using cosine distance.
    Embeddings are precomputed and stored in-memory for sub-50ms retrieval.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.vectorizer = SentenceTransformerVectorizer(model_name)
        self.embeddings: np.ndarray | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        """
        Build index by encoding all chunks into semantic embeddings.
        
        Args:
            chunks: List of Chunk objects to index
        """
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.embeddings = self.vectorizer.fit_transform(texts)
        logger.info(f"Built semantic index with {len(self.chunks)} chunks (embeddings shape: {self.embeddings.shape})")

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Semantic similarity search using cosine distance.
        
        Args:
            query: Query text
            k: Number of top results to return
            
        Returns:
            List of (Chunk, similarity_score) tuples, sorted by score descending
        """
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        
        # Encode query
        q_emb = self.vectorizer.transform([query])
        
        # Cosine similarity: embeddings already L2-normalized, so dot product = similarity
        scores = (self.embeddings @ q_emb.T).ravel()
        
        # Get top-k indices
        top_idx = np.argsort(scores)[::-1][:k]
        results: List[Tuple[Chunk, float]] = []
        for idx in top_idx:
            results.append((self.chunks[int(idx)], float(scores[int(idx)])))
        return results
