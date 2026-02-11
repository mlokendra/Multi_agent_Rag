from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List
from contract_rag.agents.retriever import Retriever
from contract_rag.config import settings
from contract_rag.agents.router import Router
from contract_rag.utils.textnorm import normalize_text
from contract_rag.backend_stub import LocalHybridBackend

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight contract RAG CLI")
    parser.add_argument("question", nargs="*", help="Optional single question to answer and exit")
    parser.add_argument(
        "--llm-provider",
        choices=["transformers", "openai", "none"],
        default=settings.llm_provider,
        help="Choose LLM backend (default from settings.llm_provider).",
    )
    return parser
@dataclass
class EvalScore:
    query: str
    reference_answer: str
    system_answer: str
    token_overlap: float  # 0-1
    keyword_coverage: float  # 0-1
    answer_length_ratio: float  # 0-1
    has_citations: bool
    evidence_sufficiency: str
    composite_score: float  # 0-1 weighted average


class DeterministicMetrics:
    """Deterministic metrics for contract QA evaluation."""

    @staticmethod
    def tokenize_simple(text: str) -> set:
        """Simple whitespace tokenization with normalization."""
        text = normalize_text(text).lower()
        tokens = set(text.split())
        return {t for t in tokens if len(t) > 2}  # Filter very short tokens

    @staticmethod
    def token_overlap(ref: str, sys: str) -> float:
        """Compute Jaccard similarity between reference and system answer."""
        ref_tokens = DeterministicMetrics.tokenize_simple(ref)
        sys_tokens = DeterministicMetrics.tokenize_simple(sys)
        
        if not ref_tokens and not sys_tokens:
            return 1.0
        if not ref_tokens or not sys_tokens:
            return 0.0
        
        intersection = len(ref_tokens & sys_tokens)
        union = len(ref_tokens | sys_tokens)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def keyword_coverage(ref: str, sys: str) -> float:
        """Coverage of reference keywords in system answer."""
        ref_tokens = DeterministicMetrics.tokenize_simple(ref)
        sys_tokens = DeterministicMetrics.tokenize_simple(sys)
        
        if not ref_tokens:
            return 1.0
        
        covered = len(ref_tokens & sys_tokens)
        return covered / len(ref_tokens)

    @staticmethod
    def answer_length_ratio(ref: str, sys: str) -> float:
        """Penalize answers that are too short or too long."""
        ref_len = len(ref.split())
        sys_len = len(sys.split())
        
        if ref_len == 0:
            return 1.0 if sys_len == 0 else 0.0
        
        ratio = sys_len / ref_len
        # Ideal ratio is 0.8-1.2 (allow some variation)
        if 0.5 <= ratio <= 2.0:
            return 1.0 - abs(1.0 - ratio) * 0.3
        else:
            return max(0.0, 1.0 - abs(1.0 - ratio) * 0.5)

    @staticmethod
    def compute_composite_score(
        token_overlap: float,
        keyword_coverage: float,
        answer_length_ratio: float,
        has_citations: bool,
        evidence_sufficiency: str,
    ) -> float:
        """Weighted composite score."""
        base_score = (
            token_overlap * 0.35 +          # Exact token match
            keyword_coverage * 0.35 +        # Reference keyword coverage
            answer_length_ratio * 0.20       # Answer length appropriateness
        )
        
        # Bonus for citations
        if has_citations:
            base_score += 0.05
        
        # Adjust by evidence sufficiency
        if evidence_sufficiency == "sufficient":
            base_score *= 1.0
        elif evidence_sufficiency == "partial":
            base_score *= 0.85
        else:  # insufficient
            base_score *= 0.6
        
        return min(1.0, base_score)


def main() -> None:
    dataset_path = Path(__file__).parent / "dataset.jsonl"
    parser = build_cli()
    args = parser.parse_args()

    retriever = Retriever(backend=LocalHybridBackend())
    router = Router(retriever=retriever, llm_provider=args.llm_provider)
    
    scores: List[EvalScore] = []
    total = 0
    
    print("\n" + "="*80)
    print("CONTRACT RAG EVALUATION")
    print("="*80 + "\n")
    
    for line in dataset_path.read_text().splitlines():
        if not line.strip():
            continue
        
        total += 1
        rec = json.loads(line)
        query = rec["query"]
        reference_answer = rec["answer"]
        
        print(f"\n[Query {total}] {query}")
        print(f"Reference: {reference_answer}")
        
        resp_text = router.handle(query)
        
        # Parse the response to extract answer and citations
        # Assuming synthesizer.render() format
        lines = resp_text.split("\n")
        system_answer = lines[0].strip() if lines else ""
        has_citations = "Citations:" in resp_text
        
        # Try to extract evidence_sufficiency from the response
        # For now, assume sufficient if we have content
        evidence_sufficiency = "sufficient" if system_answer and system_answer != "Not found in the provided contracts (no relevant evidence retrieved)." else "insufficient"
        
        # Compute metrics
        token_overlap = DeterministicMetrics.token_overlap(reference_answer, system_answer)
        keyword_coverage = DeterministicMetrics.keyword_coverage(reference_answer, system_answer)
        answer_length_ratio = DeterministicMetrics.answer_length_ratio(reference_answer, system_answer)
        composite = DeterministicMetrics.compute_composite_score(
            token_overlap, keyword_coverage, answer_length_ratio, has_citations, evidence_sufficiency
        )
        
        eval_score = EvalScore(
            query=query,
            reference_answer=reference_answer,
            system_answer=system_answer,
            token_overlap=token_overlap,
            keyword_coverage=keyword_coverage,
            answer_length_ratio=answer_length_ratio,
            has_citations=has_citations,
            evidence_sufficiency=evidence_sufficiency,
            composite_score=composite,
        )
        scores.append(eval_score)
        
        print(f"System:    {system_answer[:100]}..." if len(system_answer) > 100 else f"System:    {system_answer}")
        print(f"\nMetrics:")
        print(f"  Token Overlap:       {token_overlap:.3f}")
        print(f"  Keyword Coverage:    {keyword_coverage:.3f}")
        print(f"  Answer Length Ratio: {answer_length_ratio:.3f}")
        print(f"  Has Citations:       {has_citations}")
        print(f"  Evidence:            {evidence_sufficiency}")
        print(f"  Composite Score:     {composite:.3f}")
    
    # Summary statistics
    if scores:
        avg_token_overlap = sum(s.token_overlap for s in scores) / len(scores)
        avg_keyword_coverage = sum(s.keyword_coverage for s in scores) / len(scores)
        avg_answer_length = sum(s.answer_length_ratio for s in scores) / len(scores)
        avg_composite = sum(s.composite_score for s in scores) / len(scores)
        citation_rate = sum(1 for s in scores if s.has_citations) / len(scores)
        
        print("\n" + "="*80)
        print("SUMMARY METRICS")
        print("="*80)
        print(f"Total Queries Evaluated: {total}")
        print(f"Avg Token Overlap:       {avg_token_overlap:.3f}")
        print(f"Avg Keyword Coverage:    {avg_keyword_coverage:.3f}")
        print(f"Avg Answer Length Ratio: {avg_answer_length:.3f}")
        print(f"Citation Rate:           {citation_rate:.1%}")
        print(f"Avg Composite Score:     {avg_composite:.3f}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()