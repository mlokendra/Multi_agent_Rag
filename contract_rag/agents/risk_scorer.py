from __future__ import annotations

from dataclasses import dataclass
from typing import List

from contract_rag.agents.answerer import AnswerResult
from contract_rag.agents.retriever import EvidenceChunk


@dataclass(frozen=True)
class RiskFinding:
    title: str
    severity: str  # low|medium|high
    detail: str
    chunk_source: str  # e.g., "doc §section (chunk_id)"


class RiskScorer:
    """
    Rule-based risk scorer (fast + deterministic).
    Uses retrieved evidence chunks (not hallucinated).

    You can later upgrade to an LLM-based risk rubric agent,
    but rule-based is a solid baseline for evaluation.
    """

    def analyze(self, query: str, evidence: List[EvidenceChunk]) -> List[RiskFinding]:
        findings: List[RiskFinding] = []
        haystack = "\n".join([c.text.lower() for c in evidence])

        def cite(c: EvidenceChunk) -> str:
            return f"{c.doc_label} {c.section_path} (chunk={c.chunk_id})"

        # High severity patterns
        if "unlimited liability" in haystack or ("liability" in haystack and "uncapped" in haystack):
            c0 = evidence[0]
            findings.append(
                RiskFinding(
                    title="Potential unlimited/uncapped liability",
                    severity="high",
                    detail="Language suggests liability may be uncapped or unlimited. Review limitation of liability clause carefully.",
                    chunk_source=cite(c0),
                )
            )

        if "indemn" in haystack and ("without limit" in haystack or "unlimited" in haystack):
            c0 = evidence[0]
            findings.append(
                RiskFinding(
                    title="Broad indemnity without clear cap",
                    severity="high",
                    detail="Indemnification appears broad and may lack a clear monetary cap.",
                    chunk_source=cite(c0),
                )
            )

        # Medium severity patterns
        if "termination" in haystack and ("immediate" in haystack or "without notice" in haystack):
            c0 = evidence[0]
            findings.append(
                RiskFinding(
                    title="Termination may be immediate / short notice",
                    severity="medium",
                    detail="Termination language suggests immediate termination or limited notice period.",
                    chunk_source=cite(c0),
                )
            )

        if "breach" in haystack and ("72" in haystack or "hours" in haystack) and ("delay" in query.lower()):
            c0 = evidence[0]
            findings.append(
                RiskFinding(
                    title="Breach notification timeline risk",
                    severity="medium",
                    detail="Breach notification timeline is time-sensitive; delays may trigger contractual/regulatory issues.",
                    chunk_source=cite(c0),
                )
            )

        # Low severity patterns
        if "governing law" in haystack and ("conflict" in query.lower() or "conflicting" in query.lower()):
            c0 = evidence[0]
            findings.append(
                RiskFinding(
                    title="Possible governing law inconsistency",
                    severity="low",
                    detail="Governing law may differ across agreements; confirm consistency for enforcement and disputes.",
                    chunk_source=cite(c0),
                )
            )

        return findings

    def analyze_from_answer(self, answer: AnswerResult, evidence: List[EvidenceChunk]) -> List[RiskFinding]:
        # Optional hook if you want to incorporate answer text later.
        return self.analyze(query=answer.answer_text, evidence=evidence)
