# Contract RAG Prototype

A minimal retrieval-augmented QA loop tailored to the sample contracts in `../data`. It keeps dependencies light while exposing clear extension points for embeddings, reranking, and agent logic.

## Layout
- `app.py` – CLI entry point and chat loop
- `config.py` – default settings (chunk size, top_k, model names)
- `memory/` – tiny in-memory conversation store
- `ingestion/` – loaders for txt/pdf/docx, clause-aware chunking, vector index
- `retrieval/` – hybrid retriever plus simple reranker
- `agents/` – router, answer agent (optional OpenAI), risk highlighter
- `eval/` – sample dataset and runner
- `utils/` – text normalization and citation helpers

## Quickstart
1) (Optional) Install extras for PDF/DOCX and OpenAI:
```bash
pip install numpy pypdf python-docx openai
```
2) Run a one-off question:
```bash
python -m contract_rag.app "What are the termination rights?"
```
3) Start an interactive chat:
```bash
python -m contract_rag.app
```

## How it works
- Documents under `data/` load via `ingestion/load.py`.
- `ingestion/chunk.py` merges pseudo-sentences into ~900-char chunks with overlap.
- `ingestion/index.py` builds a TF-IDF cosine index (no heavy deps).
- `retrieval/retriever.py` blends vector score with keyword overlap and optional rerank.
- `agents/router.py` sends generic questions to `AnswerAgent` and risk-flavored ones to `RiskAgent` heuristically.
- If `OPENAI_API_KEY` is set, `AnswerAgent` uses the chat model in `config.py`; otherwise it concatenates top chunks with citations.

## Evaluation
```
python -m contract_rag.eval.run_eval
```
Reads `eval/dataset.jsonl`, runs the router, and prints answers.

## Extending
- Swap embeddings: plug a new embedder into `ingestion/index.py` or call OpenAI embeddings before indexing.
- Rerank: replace `retrieval/rerank.py` with a cross-encoder scorer.
- Risk logic: tune `RiskAgent.risk_terms` or add a proper checklist model.
