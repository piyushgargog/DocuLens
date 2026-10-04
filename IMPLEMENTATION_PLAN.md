# Implementation Plan — DocuLens

This was the original build plan. Every phase is complete. Per-release detail
is on the GitHub Releases page, and the reasoning behind each step is in
`DECISIONS.md`.

## Phases (all done)

| Phase | Deliverable | Today |
|---|---|---|
| 0 Setup | Requirements, `.env.example`, module skeleton | Entry point is `main.py` (FastAPI); dev tools are in `requirements-dev.txt` |
| 1 Loader | Page-aware PDF extraction; empty or corrupt input gives an empty result | Plus OCR for scans and `.docx`/`.txt`/`.md` loaders |
| 2 Chunker | Character sliding window that keeps page numbers | Chunks also carry the document name |
| 3 Embedder | `all-MiniLM-L6-v2`, loaded once | Module-level singleton; optional startup prefetch |
| 4 Vector store | FAISS `IndexFlatIP` on normalised vectors | One index per document |
| 5 LLM client | OpenAI-compatible call with a grounding prompt | Fenced untrusted passages, retries, streaming, provider failover |
| 6 Pipeline | `ingest` / `answer` | Plus hybrid `retrieve`, overview routing, summaries, suggestions |
| 7 UI | Upload, ask, show sources | Streamlit was replaced by FastAPI + a static frontend |
| 8 Testing | ≥5 questions incl. 1 unanswerable; two configurations compared | Done; `pytest` suite, browser e2e test and CI |
| 9 Docs | README and final review | Kept current with each release |
| 10 Enhancements | Multiple documents, history, summary | All built |

## Added beyond the plan

- Streaming answers with Stop, document scope, copy/export.
- Hybrid BM25 + embedding retrieval, measured with `retrieval_eval.py`.
- Provider failover across Groq, OpenRouter, NVIDIA and Hugging Face.
- Security hardening: CSP, rate limits behind a proxy, CSRF guard, `__Host-`
  cookie, prompt-injection fences, resource budgets, `pip-audit` and CodeQL.
- Structured logging with request ids and optional Sentry.
- Safe Markdown answers, a "thinking" panel, the "Calm Light" design, and an
  accessibility pass.
- Docker deployment on AWS EC2 behind Nginx with HTTPS.
