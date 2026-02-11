from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json

from contract_rag.agents.answerer import AnswerResult, Citation
from contract_rag.agents.legal_analyst import LegalAnalysisResult
from contract_rag.agents.risk_scorer import RiskFinding


@dataclass(frozen=True)
class ConsoleResponse:
    answer_text: str
    citations_text: str
    risks_text: str
    legal_text: str

    def render(self) -> str:
        parts = []
        if self.answer_text.strip():
            parts.append("Assistant answer:\n" + self.answer_text.strip())
        if self.legal_text.strip():
            parts.append("\nLegal analysis:\n" + self.legal_text.strip())
        if self.risks_text.strip():
            parts.append("\nRisk flags:\n" + self.risks_text.strip())
        if self.citations_text.strip():
            parts.append("\nCitations:\n" + self.citations_text.strip())
        return "\n".join(parts)


class Synthesizer:
    """
    Formats AnswerResult + RiskFinding into the console output required by the assignment.
    """

    def format(
        self,
        answer: AnswerResult,
        risks: Optional[List[RiskFinding]] = None,
        legal_analysis: Optional[LegalAnalysisResult] = None,
    ) -> ConsoleResponse:
        citations_text = self._format_citations(answer.citations)
        risks_text = self._format_risks(risks or [])
        legal_text = self._format_legal(legal_analysis)
        return ConsoleResponse(
            answer_text=answer.answer_text,
            citations_text=citations_text,
            risks_text=risks_text,
            legal_text=legal_text,
        )

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

    @staticmethod
    def _format_legal(result: Optional[LegalAnalysisResult]) -> str:
        if result is None:
            return ""
        data = Synthesizer._safe_parse_json(result.structured_json)
        if not data:
            return ""

        lines: List[str] = []

        def add(label: str, value: str) -> None:
            if value:
                lines.append(f"- {label}: {value}")

        add("Clause type", data.get("clause_type", ""))
        add("Agreement", data.get("agreement_name", ""))

        obligations = data.get("obligations") or []
        formatted_obligations = []
        for ob in obligations:
            party = ob.get("party", "")
            action = ob.get("action", "")
            cond = ob.get("conditions", "")
            deadline = ob.get("deadline", "")
            if any([party, action, cond, deadline]):
                pieces = [p for p in [party, action, cond, deadline] if p]
                formatted_obligations.append(" — ".join(pieces))
        if formatted_obligations:
            lines.append("- Obligations:")
            lines.extend([f"  • {item}" for item in formatted_obligations])

        liability = data.get("liability_structure") or {}
        if liability:
            cap_exists = liability.get("cap_exists", False)
            cap_desc = liability.get("cap_description", "")
            exceptions = liability.get("exceptions") or []
            li_parts = [f"Cap: {'Yes' if cap_exists else 'No'}"]
            if cap_desc:
                li_parts.append(cap_desc)
            if exceptions:
                li_parts.append("Exceptions: " + "; ".join(exceptions))
            add("Liability", " | ".join(li_parts))

        add("Governing law", data.get("governing_law", ""))
        add("Survival", data.get("survival_terms", ""))

        risk_signals = data.get("risk_signals") or []
        if risk_signals:
            lines.append("- Risk signals:")
            lines.extend([f"  • {sig}" for sig in risk_signals if sig])

        if not lines:
            return ""

        confidence = result.confidence or "unknown"
        note = f" (note: {result.note})" if result.note else ""
        lines.append(f"- Confidence: {confidence}{note}")
        return "\n".join(lines)

    @staticmethod
    def _safe_parse_json(payload: str) -> Dict[str, Any]:
        try:
            return json.loads(payload)
        except Exception:
            return {}
