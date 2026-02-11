from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b-instruct"
    temperature: float = 0.1
    seed: int = 42
    num_predict: int = 512  # response length cap


class OllamaClient:
    """
    Minimal Ollama chat client (no external deps).
    Requires: ollama running locally.

    Install + run:
      - install Ollama
      - ollama pull llama3.1:8b-instruct   (or any model you like)
      - ollama serve
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or self._load_from_env()

    @staticmethod
    def _load_from_env() -> OllamaConfig:
        return OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            seed=int(os.getenv("OLLAMA_SEED", "42")),
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
        )

    def complete(self, system: str, user: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed,
                "num_predict": self.config.num_predict,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.config.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Ollama request failed. Is Ollama running at {self.config.base_url}? Error: {e}"
            ) from e

        # Ollama returns: {"message": {"role": "...", "content": "..."} ...}
        return str(out.get("message", {}).get("content", "")).strip()
