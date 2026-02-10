from __future__ import annotations

import json
from pathlib import Path

from contract_rag.agents.router import Router


def main() -> None:
    dataset_path = Path(__file__).parent / "dataset.jsonl"
    router = Router()
    total = 0
    for line in dataset_path.read_text().splitlines():
        total += 1
        rec = json.loads(line)
        query = rec["query"]
        print(f"\nQ: {query}")
        resp = router.handle(query)
        print(f"A: {resp}")
    print(f"\nEvaluated {total} queries.")


if __name__ == "__main__":
    main()
