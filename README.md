# Contract RAG Prototype 🔍📄

A lightweight, agentic retrieval-augmented QA loop for the sample contracts in `data/`, supporting local Hugging Face models (default: **Qwen/Qwen2.5-3B-Instruct**) and OpenAI chat completions.

## Quickstart 🚀
1) Install core dependencies (virtualenv recommended):
```bash
pip install numpy pypdf python-docx openai "torch>=2.1" "transformers>=4.39" "accelerate"
```
2) Ask a one-off question with OpenAI (set `OPENAI_API_KEY` first):
```bash
python -m contract_rag.app --llm-provider openai "What are the termination rights?"
```
3) Start an interactive chat with OpenAI:
```bash
python -m contract_rag.app --llm-provider openai
```

## Layout 🗂️
- `app.py` – CLI entry point and chat loop (select LLM provider at runtime)
- `config.py` – defaults for chunking, retrieval, LLM provider/model, risk toggle
- `ingestion/` – loaders, heading-aware chunking, simple vector index
- `retrieval/` – hybrid TF-IDF/BM25-style retrieval and rerank stub
- `agents/` – router, answerer, risk scorer, synthesizer
- `llm/` – transformers client for local Qwen, optional Ollama stub, OpenAI client in `answerer`
- `eval/` – sample dataset and runner
- `utils/` – text normalization and citation helpers

### Agents 🤖
- Router: orchestrates retrieval, answering, risk scoring, and legal analysis.
- Answerer: grounded Q&A (prefers local HF model, can use OpenAI, falls back to extractive snippets).
- Legal Analyst: converts clauses into structured legal meaning and presents only populated fields.
- Risk scorer: rule-based risk flags on top retrieved evidence.
- Synthesizer: formats output as Assistant answer → Legal analysis → Risk flags → Citations.

## Architecture 🧠

The system follows a multi-agent RAG architecture with clear separation of concerns:

User Query  
→ Router Agent  
→ Retriever (hybrid TF-IDF + vector)  
→ Reranker (stub / reciprocal rank fusion)  
→ Answerer (LLM-grounded response)  
→ Risk Scorer  
→ Synthesizer → CLI Output

### Agents

- **Router** – Determines intent (QA vs risk_scan) using rule-based classification.
- **Retriever** – Retrieves heading-aware clause chunks from indexed contracts.
- **Reranker** – Reorders candidates using reciprocal-rank fusion.
- **Answerer** – Generates grounded responses using either local HF model or OpenAI.
- **Risk Scorer** – Applies rule-based legal risk heuristics.
- **Synthesizer** – Formats answer, citations, and risks for CLI display.

This separation improves modularity, testability, and grounded reasoning.


### Option A: Local Hugging Face model (default) 🤗

Environment (already in `.env`):
```
LLM_PROVIDER=transformers
HF_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
HF_MAX_NEW_TOKENS=200
HF_TEMPERATURE=0.1
HF_TOP_P=0.9
```

### Option B: OpenAI API 🤖
Environment:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Running ▶️
- One-off question (provider from env):
```bash
python -m contract_rag.app "What are the termination rights?"
```
- Force a provider at runtime:
```bash
python -m contract_rag.app --llm-provider transformers "Summarize termination"
python -m contract_rag.app --llm-provider openai "Summarize termination"
python -m contract_rag.app --llm-provider none   "Summarize termination"  # extractive fallback
```
- Interactive chat:
```bash
python -m contract_rag.app
```

## Behavior 📑
- Retrieval: heading-aware clause chunks + hybrid vector/BM25 with reciprocal-rank fusion.
- Assistant answer shown first, followed by Legal analysis (if populated), Risk flags, then Citations.
- Risk flags: run on every query by default (`RAG_ALWAYS_RISK=true`); shows severity-tagged findings under “Risk flags”.
- Legal Analyst Agent: structures clauses into JSON (clause type, obligations, liability, governing law, survival, risk signals) and renders only populated fields. Toggle with `RAG_ENABLE_LEGAL_ANALYST=true|false`.
- Citations: show doc + section for each chunk used.

## Flow 🪄
1. Router detects intent (QA vs risk scan) and reuses retrieval for all agents.
2. Retrieval + rerank gather the top evidence chunks.
3. Answerer generates the grounded reply (or extractive fallback).
4. Legal Analyst (if enabled) structures obligations/liability/governing law/survival from the same evidence.
5. Risk scorer flags heuristic risks.
6. Synthesizer orders output: Assistant answer → Legal analysis → Risk flags → Citations.

## Evaluation 🧪
```bash
python -m contract_rag.eval.run_eval
```
Reads `eval/dataset.jsonl`, runs the pipeline, and prints answers.

## Configuration Knobs 🔧
- LLM provider/model: `.env` or `--llm-provider`.
- Risk scoring toggle: `RAG_ALWAYS_RISK=true|false`.
- Chunking/overlap/top_k: see `config.py`.

## Future Improvements 🚀

- Cross-document conflict detection agent
- Learned reranker (e.g., bge-reranker)
- LLM-based risk scoring with structured output
- Persistent vector database
- Citation confidence scoring


## Notes 📝
- If the local model cannot load (e.g., missing weights/GPU), the app logs a warning and falls back to extractive snippets.
- OpenAI quota errors will trigger the fallback; check your billing/quota if that happens.
