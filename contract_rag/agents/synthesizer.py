from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from contract_rag.agents.answerer import AnswerResult, Citation
from contract_rag.agents.risk_scorer import RiskFinding


@dataclass(frozen=True)
class ConsoleResponse:
    answer_text: str
    citations_text: str
    risks_text: str

    def render(self) -> str:
        parts = [self.answer_text.strip()]
        if self.citations_text.strip():
            parts.append("\nCitations:\n" + self.citations_text.strip())
        if self.risks_text.strip():
            parts.append("\nRisk flags:\n" + self.risks_text.strip())
        return "\n".join(parts)


class Synthesizer:
    """
    Formats AnswerResult + RiskFinding into the console output required by the assignment.
    """

    def format(
        self,
        answer: AnswerResult,
        risks: Optional[List[RiskFinding]] = None,
    ) -> ConsoleResponse:
        citations_text = self._format_citations(answer.citations)
        risks_text = self._format_risks(risks or [])
        return ConsoleResponse(answer_text=answer.answer_text, citations_text=citations_text, risks_text=risks_text)

    @staticmethod
    def _format_citations(citations: List[Citation]) -> str:
        if not citations:
            return ""
        lines = []
        for c in citations:
            lines.append(f"- {c.citation_id}: {c.doc_label} {c.section_path} (chunk={c.chunk_id}) — “{c.excerpt}…”")
        return "\n".join(lines)

    @staticmethod
    def _format_risks(risks: List[RiskFinding]) -> str:
        if not risks:
            return ""
        lines = []
        for r in risks:
            lines.append(f"- [{r.severity.upper()}] {r.title}: {r.detail} | evidence: {r.chunk_source}")
        return "\n".join(lines)
