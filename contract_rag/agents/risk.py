from __future__ import annotations

from dataclasses import dataclass
from typing import List

from contract_rag.retrieval.retriever import HybridRetriever


@dataclass
class RiskFinding:
    title: str
    detail: str
    chunk_source: str


class RiskAgent:
    """Very lightweight risk highlighter using keyword heuristics + retrieval."""

    risk_terms = {
        "liability": "Check liability caps and exclusions.",
        "indemn": "Indemnification scope or carve-outs detected.",
        "termination": "Termination rights/notice period referenced.",
        "confidential": "Confidentiality obligations found.",
        "sla": "Service levels or remedies mentioned.",
    }

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def analyze(self, question: str) -> List[RiskFinding]:
        results = self.retriever.retrieve(question, k=8)
        findings: List[RiskFinding] = []
        for res in results:
            text_lower = res.chunk.text.lower()
            for term, hint in self.risk_terms.items():
                if term in text_lower:
                    findings.append(
                        RiskFinding(
                            title=term.capitalize(),
                            detail=f"{hint} Example: {res.chunk.text[:180]}...",
                            chunk_source=res.chunk.source,
                        )
                    )
        return findings
