from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import logging
import re

from contract_rag.agents.answerer import LLMClient, OpenAIChatClient
from contract_rag.agents.reranker import Reranker
from contract_rag.agents.retriever import EvidenceChunk, Retriever
from contract_rag.config import settings
from contract_rag.llm.transformers_client import TransformersClient, TransformersConfig


@dataclass(frozen=True)
class Obligation:
    party: str
    action: str
    conditions: str
    deadline: str


@dataclass(frozen=True)
class LiabilityStructure:
    cap_exists: bool
    cap_description: str
    exceptions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalAnalysisResult:
    structured_json: str
    confidence: str  # low|medium|high
    note: str = ""


class LegalAnalyst:
    """
    Interprets retrieved contract clauses and converts them into structured legal meaning.
    Prefers an LLM; falls back to a deterministic placeholder if none is configured.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLMClient] = None,
        preferred_provider: Optional[str] = None,
        top_k: int = 8,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.top_k = top_k
        self.llm = llm or self._init_llm(preferred_provider)

    def analyze(self, query: str, evidence: Optional[List[EvidenceChunk]] = None) -> LegalAnalysisResult:
        evidence = evidence or self._retrieve(query)
        if not evidence:
            return LegalAnalysisResult(
                structured_json="{}",
                confidence="low",
                note="No evidence available for legal analysis.",
            )

        if self.llm is None:
            return LegalAnalysisResult(
                structured_json=self._fallback_json(evidence),
                confidence="low",
                note="LLM not configured; returned minimal heuristic mapping.",
            )

        system_prompt = (
            "You are a Legal Analyst Agent. Interpret retrieved contract clauses and output structured JSON.\n"
            "Follow these rules strictly:\n"
            "- Use ONLY the provided clauses; do not invent terms.\n"
            "- If a field is unknown, use an empty string or empty array.\n"
            "- Keep values concise; avoid repetition of full clauses.\n"
        )

        schema = (
            '{\n'
            '  "agreement_name": "",\n'
            '  "clause_type": "",\n'
            '  "obligations": [\n'
            '    {"party": "", "action": "", "conditions": "", "deadline": ""}\n'
            '  ],\n'
            '  "liability_structure": {"cap_exists": true, "cap_description": "", "exceptions": []},\n'
            '  "governing_law": "",\n'
            '  "survival_terms": "",\n'
            '  "risk_signals": []\n'
            '}'
        )

        example = (
            'Example mapping:\n'
            'Clause: "Confidentiality obligations shall survive termination for five (5) years."\n'
            'JSON:\n'
            '{\n'
            '  "clause_type": "Survival",\n'
            '  "obligations": [{"party": "Receiving Party", "action": "Maintain confidentiality", '
            '"conditions": "After termination", "deadline": "5 years after termination"}],\n'
            '  "survival_terms": "Confidentiality survives 5 years post termination",\n'
            '  "risk_signals": ["Medium: long survival period"]\n'
            '}'
        )

        evidence_block = self._format_evidence_for_prompt(evidence)
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Return JSON following this schema:\n"
            f"{schema}\n\n"
            f"{example}\n\n"
            "Now produce the JSON for this question and evidence."
        )

        try:
            raw = self.llm.complete(system=system_prompt, user=user_prompt).strip()
            structured = self._extract_json_block(raw)
            return LegalAnalysisResult(structured_json=structured, confidence="medium")
        except Exception as exc:
            logging.getLogger(__name__).warning("Legal analyst LLM call failed: %s", exc)
            return LegalAnalysisResult(
                structured_json=self._fallback_json(evidence),
                confidence="low",
                note="LLM call failed; returned heuristic mapping.",
            )

    def _retrieve(self, query: str) -> List[EvidenceChunk]:
        chunks = self.retriever.retrieve(query, top_k=max(self.top_k, 10))
        return self.reranker.rerank(query, chunks)[: self.top_k]

    @staticmethod
    def _format_evidence_for_prompt(chunks: List[EvidenceChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[E{i}] doc={c.doc_label} section={c.section_path} chunk={c.chunk_id}\n{c.text.strip()}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_json(chunks: List[EvidenceChunk]) -> str:
        first = chunks[0]
        return (
            '{\n'
            f'  "clause_type": "unknown",\n'
            f'  "obligations": [{{"party": "", "action": "{first.text[:120].strip()}", "conditions": "", "deadline": ""}}],\n'
            f'  "liability_structure": {{"cap_exists": false, "cap_description": "", "exceptions": []}},\n'
            f'  "governing_law": "",\n'
            f'  "survival_terms": "",\n'
            f'  "risk_signals": ["LLM unavailable"]\n'
            '}\n'
        )

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
                logger.info("LegalAnalyst LLM provider=transformers model=%s", cfg.model_name)
                return TransformersClient(cfg)
            except Exception as exc:  # pragma: no cover
                logger.warning("Transformers LLM setup failed for LegalAnalyst: %s", exc)

        if provider == "openai" and settings.openai_api_key:
            try:
                logger.info("LegalAnalyst LLM provider=openai model=%s", settings.openai_chat_model)
                return OpenAIChatClient(settings.openai_chat_model, settings.openai_api_key)
            except Exception as exc:  # pragma: no cover
                logger.warning("OpenAI LLM setup failed for LegalAnalyst: %s", exc)

        logger.info("LegalAnalyst LLM not configured; will use heuristic fallback.")
        return None

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Try to pull the first JSON object out of the model response,
        stripping markdown fences if present.
        """
        if not text:
            return "{}"

        # Remove code fences like ```json ... ```
        fence_pattern = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)
        match = fence_pattern.search(text)
        if match:
            text = match.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()
        return text.strip()
