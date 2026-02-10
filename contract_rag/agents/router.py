from __future__ import annotations

from dataclasses import dataclass

from contract_rag.agents.answer import AnswerAgent
from contract_rag.agents.risk import RiskAgent


@dataclass
class RouterDecision:
    route: str
    reason: str


class Router:
    def __init__(self, answer_agent: AnswerAgent | None = None, risk_agent: RiskAgent | None = None) -> None:
        self.answer_agent = answer_agent or AnswerAgent()
        self.risk_agent = risk_agent or RiskAgent(self.answer_agent.retriever)

    @staticmethod
    def _is_risk_query(query: str) -> bool:
        q = query.lower()
        triggers = ["risk", "liability", "indemn", "breach", "termination", "penalty"]
        return any(t in q for t in triggers)

    def route(self, query: str) -> RouterDecision:
        if self._is_risk_query(query):
            return RouterDecision(route="risk", reason="Risk-related keywords present")
        return RouterDecision(route="answer", reason="Default factual Q&A")

    def handle(self, query: str) -> str:
        decision = self.route(query)
        if decision.route == "risk":
            findings = self.risk_agent.analyze(query)
            if not findings:
                return "No specific risks detected in the retrieved context."
            lines = [f"- {f.title}: {f.detail} (source: {f.chunk_source})" for f in findings]
            return "\n".join(lines)
        return self.answer_agent.answer(query)
