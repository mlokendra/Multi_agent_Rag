"""Central configuration for the contract RAG prototype.

This module keeps defaults small and dependency-light so the project can run
inside the provided sandbox. Environment variables let you swap components
without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_fallback(env_path: Path) -> None:
    """Minimal .env loader (KEY=VALUE) used when python-dotenv is absent."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


# Optional: load environment variables from project .env if present.
env_file = Path(__file__).resolve().parent.parent / ".env"
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(env_file)
except Exception:
    _load_env_fallback(env_file)


@dataclass
class Settings:
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    # Chunking
    target_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", 900))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", 120))
    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", 3))
    hybrid_weight: float = float(os.getenv("RAG_HYBRID_WEIGHT", 0.35))
    # Embeddings / LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "transformers")  # transformers | openai | none
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    hf_model_name: str = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    hf_max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "200"))
    hf_temperature: float = float(os.getenv("HF_TEMPERATURE", "0.1"))
    hf_top_p: float = float(os.getenv("HF_TOP_P", "0.9"))
    # Risk
    always_compute_risks: bool = os.getenv("RAG_ALWAYS_RISK", "true").lower() == "true"
    # Paths
    cache_dir: Path = Path(os.getenv("RAG_CACHE_DIR", Path(__file__).resolve().parent / "tmp"))


settings = Settings()
