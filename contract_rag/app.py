from __future__ import annotations

import argparse
import logging

from contract_rag.agents.router import Router
from contract_rag.agents.retriever import Retriever
from contract_rag.backend_stub import LocalHybridBackend
from contract_rag.config import settings
from contract_rag.memory.store import MemoryStore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    memory = MemoryStore()
    retriever = Retriever()#(backend=LocalHybridBackend())
    # Pass provider choice down to Answerer via Router
    router = Router(retriever=retriever, llm_provider=args.llm_provider)

    # Log whether LLM is active for this session.
    if getattr(router.answerer, "llm", None):
        logging.info(
            "LLM connected for answering (provider=%s model=%s).",
            settings.llm_provider,
            settings.hf_model_name if settings.llm_provider == "transformers" else settings.openai_chat_model,
        )
    else:
        logging.info("LLM not configured; fallback to extractive snippets.")

    if args.question:
        query = " ".join(args.question)
        memory.add_user(query)
        print(router.handle(query))
        return

    print("Contract RAG ready. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("bye")
            break
        memory.add_user(query)
        answer = router.handle(query)
        memory.add_assistant(answer)
        print(f"assistant> {answer}\n")


if __name__ == "__main__":
    main()
