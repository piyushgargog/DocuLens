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

# v4 upgrade plan (built, being hardened; not released)

Written 2026-10-05 from a read of the source, not from the docs. Each phase
ends with the full test suite green, the app booting, and `DECISIONS.md` /
`ARCHITECTURE.md` updated. Nothing is released until the owner says so.

## What is already solved, verified in code

| Area | State |
|---|---|
| Identity | Firebase ID tokens verified server-side (`auth.py`); our own HttpOnly cookie + 7-day store entry. Guests exist by design (1 doc, 5 questions/day). |
| Shared state | Upstash Redis via `store.py` (`login:*`, `rl:*`), Lua sliding-window limiter checked against the live database. Falls back to process memory if Redis fails. |
| Documents | **Not persistent.** `main.Session.docs` is a process-local dict of `pipeline.IndexState` (text, embeddings, BM25 stats, FAISS flat index). Lost on restart, invisible to a second instance. History is the same. |
| Chunking | Fixed 800/150 character window per page (`chunker.py`). Tried a sentence-aware variant before; reverted after it lowered Hybrid Hit@4 (0.82 → 0.79). |
| Summary | 10 evenly spaced chunks, one LLM call (`pipeline.summarize`). |
| Follow-ups | Search `"<previous question> <question>"` as a second query; last 3 turns sent as untrusted reference. |
| Vector search | One `IndexFlatIP` per document plus a NumPy matrix product in `retrieve` (the FAISS index itself is not used by the hybrid path). |
| OCR | Whole-PDF fallback only when *no* page has a text layer; 30 pages, 144 DPI, no preprocessing or rotation handling. |

## Phases

1. **Foundations**: map the code, write this plan, confirm Firebase and
   Upstash work against the live service. Done.
2. **Persistent, shared documents.** `docstore.py`: a `DocumentRepository`
   (metadata, ownership, history in Redis under `u:<uid>:*`) over an
   `ArtifactStore` (compressed chunks + embeddings; backends: Redis blobs,
   local disk, in-memory). Signed-in users' documents survive restarts and
   resolve on any instance; guests stay in memory. Lazy, bounded in-process
   cache; delete removes metadata, artifacts and history.
3. **Vector index abstraction.** `VectorIndex` (add/search/save/load/remove/
   size) with exact `IndexFlatIP` and HNSW; switch point chosen from a
   benchmark, not assumed.
4. **Chunking + evaluation.** Structure-aware chunker, benchmarked against the
   current one on `retrieval_eval.py` (extended with paraphrase, number, name,
   cross-page, table, short and follow-up questions). Shipped only if it wins.
5. **Conversational retrieval.** Deterministic reference resolver first, an
   explicit separate LLM rewrite only when needed, concatenation as fallback.
   The rewrite is a search query, never evidence.
6. **Hierarchical summary.** Batch by section/page range, bounded LLM budget,
   page ranges carried through every level.
7. **OCR.** Per-page need detection, orientation, preprocessing, per-page
   failure isolation, configurable limits.

## Constraints that shape the design

- The EC2 box has 2 GB RAM and ~680 MB free disk: no local vector DB, no big
  local artifact store. Redis blobs are bounded per user; an S3-compatible
  backend can be added behind `ArtifactStore` later.
- Upstash free tier is limited in storage and commands, so state reads must be
  few (one hash read per request) and artifacts loaded lazily.
- Free-tier LLM quotas: every added LLM call (rewrite, map-reduce) needs an
  explicit budget and a deterministic fallback.

## Status

Phases 2-7 are built and tested (480 tests, 10 browser tests). Measurements and
the decisions they drove are in `DECISIONS.md` (v4 entry) and `reports/`.
Remaining before release: the security hardening pass, deployment to the AWS
host for the owner's check, then the release.
