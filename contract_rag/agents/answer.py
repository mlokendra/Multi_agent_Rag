from __future__ import annotations

import os
from typing import List

from contract_rag.config import settings
from contract_rag.retrieval.retriever import HybridRetriever, RetrievedChunk
from contract_rag.utils.citations import Citation, attach_citations

try:  # optional dependency
    import openai
except Exception:  # pragma: no cover
    openai = None


class AnswerAgent:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()
        if openai and settings.openai_api_key:
            openai.api_key = settings.openai_api_key

    @staticmethod
    def _format_context(results: List[RetrievedChunk]) -> str:
        lines = []
        for i, res in enumerate(results):
            lines.append(f"[{i+1}] {res.chunk.text}\n-- {res.chunk.source}")
        return "\n\n".join(lines)

    def _llm_answer(self, question: str, results: List[RetrievedChunk]) -> str | None:
        if not openai or not settings.openai_api_key:
            return None
        context = self._format_context(results)
        prompt = (
            "You are a contract analyst. Answer the question using ONLY the provided context. "
            "Include short citations like [1], [2] matching the context blocks.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        try:
            resp = openai.ChatCompletion.create(
                model=settings.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            return resp.choices[0].message["content"].strip()
        except Exception:
            return None

    @staticmethod
    def _extract_citations(results: List[RetrievedChunk]) -> List[Citation]:
        cits: List[Citation] = []
        for res in results:
            cits.append(Citation(chunk_id=res.chunk.id, source=res.chunk.source, text=res.chunk.text, score=res.score))
        return cits

    def answer(self, question: str) -> str:
        results = self.retriever.retrieve(question)
        llm_out = self._llm_answer(question, results)
        if llm_out:
            return attach_citations(llm_out, self._extract_citations(results))

        # Deterministic fallback: stitch together the top chunks.
        top_texts = [f"[{i+1}] {res.chunk.text}" for i, res in enumerate(results)]
        answer = "\n".join(top_texts)
        return attach_citations(answer, self._extract_citations(results))
