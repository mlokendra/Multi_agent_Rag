from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol
import logging

from contract_rag.agents.retriever import EvidenceChunk, Retriever
from contract_rag.agents.reranker import Reranker
from contract_rag.config import settings
from contract_rag.llm.transformers_client import TransformersClient, TransformersConfig

try:  # Optional OpenAI dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass(frozen=True)
class Citation:
    citation_id: str
    doc_label: str
    section_path: str
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class AnswerResult:
    answer_text: str
    citations: List[Citation]
    evidence_sufficiency: str  # sufficient|partial|insufficient


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIChatClient:
    """OpenAI chat client (SDK >=1.0)."""

    def __init__(self, model: str, api_key: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not installed; cannot use OpenAI provider.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()


class Answerer:
    """
    Grounded answering agent (local-first).
    - Uses local HF model by default (Qwen/Qwen2.5-3B-Instruct).
    - Falls back to extractive answer if the model is unavailable.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLMClient] = None,
        top_k: int = 8,
        preferred_provider: Optional[str] = None,  # "transformers", "openai", or None
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.top_k = top_k

        self.llm = llm or self._init_llm(preferred_provider)
            

    def answer(self, query: str, filters: dict | None = None) -> AnswerResult:
        chunks = self.retriever.retrieve(query, top_k=max(self.top_k, 10), filters=filters)
        chunks = self.reranker.rerank(query, chunks)[: self.top_k]

        if not chunks:
            return AnswerResult(
                answer_text="Not found in the provided contracts (no relevant evidence retrieved).",
                citations=[],
                evidence_sufficiency="insufficient",
            )

        citations = self._build_citations(chunks)

        # If LLM fails (e.g., model not available), fallback safely.
        if self.llm is None:
            return self._extractive_fallback(chunks, citations)

        system_prompt = (
            "You are a legal contract QA assistant.\n"
            "Rules:\n"
            "1) Answer ONLY using the provided evidence.\n"
            "2) Do NOT guess; if missing, say 'Not found in provided contracts'.\n"
            "3) Keep it concise.\n"
        )
        evidence_block = self._format_evidence_for_prompt(chunks)
        user_prompt = f"Question:\n{query}\n\nEvidence:\n{evidence_block}\n\nAnswer:"

        try:
            text = self.llm.complete(system=system_prompt, user=user_prompt).strip()
            return AnswerResult(answer_text=text, citations=citations, evidence_sufficiency="sufficient")
        except Exception:
            return self._extractive_fallback(chunks, citations)

    def _init_llm(self, preferred_provider: Optional[str]) -> Optional[LLMClient]:
        logger = logging.getLogger(__name__)
        provider = (preferred_provider or settings.llm_provider).lower()

        if provider == "transformers":
            try:
                cfg = TransformersConfig(
                    model_name=settings.hf_model_name,
                    max_new_tokens=settings.hf_max_new_tokens,
                    temperature=settings.hf_temperature,
                    top_p=settings.hf_top_p,
                )
                logger.info("LLM provider=transformers model=%s", cfg.model_name)
                return TransformersClient(cfg)
            except Exception as exc:
                logger.warning("Transformers LLM setup failed: %s", exc)

        if provider == "openai" and settings.openai_api_key:
            try:
                logger.info("LLM provider=openai model=%s", settings.openai_chat_model)
                return OpenAIChatClient(settings.openai_chat_model, settings.openai_api_key)
            except Exception as exc:
                logger.warning("OpenAI LLM setup failed: %s", exc)

        logger.info("LLM not configured; using extractive fallback.")
        return None

    @staticmethod
    def _extractive_fallback(chunks: List[EvidenceChunk], citations: List[Citation]) -> AnswerResult:
        best = chunks[0]
        answer = (
            "I couldn't reach the local LLM, so here is the most relevant retrieved clause text:\n\n"
            f"{best.text.strip()}\n"
        )
        return AnswerResult(answer_text=answer, citations=citations, evidence_sufficiency="partial")

    @staticmethod
    def _format_evidence_for_prompt(chunks: List[EvidenceChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[E{i}] doc={c.doc_label} section={c.section_path} chunk={c.chunk_id}\n{c.text.strip()}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _build_citations(chunks: List[EvidenceChunk]) -> List[Citation]:
        citations: List[Citation] = []
        for i, c in enumerate(chunks, start=1):
            excerpt = " ".join(c.text.strip().split())[:180]
            citations.append(
                Citation(
                    citation_id=f"C{i}",
                    doc_label=c.doc_label,
                    section_path=c.section_path,
                    chunk_id=c.chunk_id,
                    excerpt=excerpt,
                )
            )
        return citations
    
