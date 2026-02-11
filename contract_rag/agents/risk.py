from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

from contract_rag.retrieval.retriever import HybridRetriever


@dataclass
class RiskFinding:
    title: str
    detail: str
    chunk_source: str
    severity: str


class RiskAgent:
    """Very lightweight risk highlighter using keyword heuristics + retrieval."""

    risk_terms: Dict[str, Dict[str, str]] = {
        "liability": {"hint": "Check liability caps and exclusions.", "severity": "high"},
        "indemn": {"hint": "Indemnification scope or carve-outs detected.", "severity": "high"},
        "termination": {"hint": "Termination rights/notice period referenced.", "severity": "medium"},
        "confidential": {"hint": "Confidentiality obligations found.", "severity": "medium"},
        "sla": {"hint": "Service levels or remedies mentioned.", "severity": "medium"},
    }

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def analyze(self, question: str) -> List[RiskFinding]:
        results = self.retriever.retrieve(question, k=8)
        findings: List[RiskFinding] = []
        for res in results:
            text_lower = res.chunk.text.lower()
            for term, info in self.risk_terms.items():
                if term in text_lower:
                    findings.append(
                        RiskFinding(
                            title=term.capitalize(),
                            detail=f"{info['hint']} Example: {res.chunk.text[:180]}...",
                            chunk_source=res.chunk.source,
                            severity=info["severity"],
                        )
                    )
        return findings
