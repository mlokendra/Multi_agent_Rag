"""Central configuration for the contract RAG prototype.

This module keeps defaults small and dependency-light so the project can run
inside the provided sandbox. Environment variables let you swap components
without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    # Chunking
    target_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", 900))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", 120))
    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", 6))
    hybrid_weight: float = float(os.getenv("RAG_HYBRID_WEIGHT", 0.35))
    # Embeddings / LLM
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    # Paths
    cache_dir: Path = Path(os.getenv("RAG_CACHE_DIR", Path(__file__).resolve().parent / "tmp"))


settings = Settings()
