from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from contract_rag.agents.answerer import Answerer
from contract_rag.agents.legal_analyst import LegalAnalyst
from contract_rag.agents.retriever import Retriever
from contract_rag.agents.reranker import Reranker
from contract_rag.agents.risk_scorer import RiskScorer
from contract_rag.agents.synthesizer import Synthesizer
from contract_rag.config import settings

@dataclass
class RouterDecision:
    route: str
    reason: str
    intent: "RouterIntent"


@dataclass
class RouterIntent:
    intent: str  # qa, clause_lookup, compare_conflicts, risk_scan, summarize_risks
    targets: List[str] = field(default_factory=lambda: ["all_docs"])
    constraints: Dict[str, bool | str] = field(default_factory=dict)


class Router:
    """
    Thin orchestration layer:
      - determines intent
      - executes retrieval + answer
      - optionally runs risk scoring
      - formats output

    This stays deterministic and testable.
    """

    def __init__(
        self,
        retriever: Retriever,
        answerer: Optional[Answerer] = None,
        reranker: Optional[Reranker] = None,
        risk_scorer: Optional[RiskScorer] = None,
        legal_analyst: Optional[LegalAnalyst] = None,
        synthesizer: Optional[Synthesizer] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker or Reranker()
        self.answerer = answerer or Answerer(
            retriever=self.retriever,
            reranker=self.reranker,
            preferred_provider=llm_provider,
        )
        self.risk_scorer = risk_scorer or RiskScorer()
        self.legal_analyst = legal_analyst or LegalAnalyst(
            retriever=self.retriever,
            reranker=self.reranker,
            preferred_provider=llm_provider,
        )
        self.synthesizer = synthesizer or Synthesizer()

    @staticmethod
    def _is_risk_query(query: str) -> bool:
        q = query.lower()
        # Expanded triggers include simple synonyms and plural/non-stem variants
        triggers = [
            "risk", "exposure", "hazard",
            "liability", "liabilities", "uncapped", "unlimited", "cap", "capped",
            "indemn", "indemnify", "indemnity", "hold harmless",
            "breach", "violation",
            "termination", "terminate", "terminating", "cancel", "cancellation", "notice period",
            "penalty", "penalties", "fine",
            "damages", "loss", "losses",
            "service credit", "uptime", "sla",
            "data breach", "security incident",
        ]
        return any(t in q for t in triggers)

    def route(self, query: str) -> RouterDecision:
        if self._is_risk_query(query):
            intent = RouterIntent(intent="risk_scan", targets=["all_docs"], constraints={"risk": True})
            return RouterDecision(route="risk", reason="Risk-related keywords present", intent=intent)

        intent = RouterIntent(intent="qa", targets=["all_docs"], constraints={"risk": False})
        return RouterDecision(route="answer", reason="Default factual Q&A", intent=intent)

    def handle(self, query: str) -> str:
        decision = self.route(query)

        # Retrieve evidence once; reuse for answer + risk (important for efficiency & grounding)
        evidence = self.retriever.retrieve(query, top_k=30)
        evidence = self.reranker.rerank(query, evidence)

        answer = self.answerer.answer(query)

        risks = []
        if decision.intent.constraints.get("risk") is True or settings.always_compute_risks:
            risks = self.risk_scorer.analyze(query=query, evidence=evidence[:10])

        legal_analysis = None
        if settings.enable_legal_analyst:
            legal_analysis = self.legal_analyst.analyze(query=query, evidence=evidence[:8])

        output = self.synthesizer.format(answer=answer, risks=risks, legal_analysis=legal_analysis)
        return output.render()
