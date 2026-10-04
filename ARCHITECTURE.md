# Architecture — DocuLens

_Current as of v3.7.0._

## Overview

One FastAPI process serves the API and a static HTML/CSS/JS frontend from the
same origin (no CORS). The RAG pipeline is a set of plain Python functions,
with no LangChain, so every stage can be read and tested on its own. The
history behind each choice is in `DECISIONS.md`.

## Components

| Module | Responsibility |
|---|---|
| `main.py` | API, sessions, rate limits, concurrency, security middleware. The only file that imports FastAPI. |
| `document_loader.py`, `pdf_loader.py` | Pick a loader by extension. PDFs are read per page with PyMuPDF, and a PDF with no text layer is OCR'd (Tesseract, ≤30 pages, ~144 DPI). `.txt`/`.md`/`.docx` are split into ~2500-character pseudo-pages. |
| `chunker.py` | Character sliding window per page (800/150). Each chunk keeps its page. |
| `embedder.py` | `all-MiniLM-L6-v2`, L2-normalised 384-dim vectors. Loaded once per process, at startup when `PREFETCH_MODEL=1`. |
| `vector_store.py` | One FAISS `IndexFlatIP` per document (inner product = cosine). |
| `retriever.py` | BM25 and reciprocal rank fusion. |
| `pipeline.py` | `ingest`, `retrieve`, `gather_sources`, `answer`, `answer_stream`, `summarize`, `suggest_questions`. |
| `llm_client.py`, `providers.py` | Prompts, OpenAI-compatible calls over `requests`, streaming, provider failover. |
| `observability.py` | `doculens` logger, request ids, optional Sentry. |
| `static/` | Frontend, with no framework or build step. |

## Data flow

```
POST /api/ingest → load pages → chunk → embed → IndexState (FAISS + chunks)
                 → session.docs[doc_id]
POST /api/ask(/stream) → gather_sources
     overview question → ordered sample (first 2 chunks + evenly spaced, 8 total)
     otherwise         → BM25 + cosine rankings per query, fused (RRF, k=60) → top 4
   → prompt: system rules + fenced passages + last 3 turns + question
   → provider chain → answer + sources
```

## Sessions and state

- Sessions are held in an in-memory `dict[str, Session]` keyed by a random
  256-bit cookie (`__Host-session` over HTTPS). Each session holds up to 5
  documents and the last 10 turns.
- Sessions expire after 2 hours idle; a background sweep runs every 5 minutes.
  At most 50 sessions exist; the oldest idle one is evicted first. Active API
  responses slide the cookie's expiry forward without changing the id.
- Each session has an `RLock`. Ingestion, answering and the sweeper run on
  different threads.
- `MAX_TOTAL_CHUNKS` (75,000) caps indexed chunks across all sessions. It is
  checked before embedding and again before storing. A new session is
  registered only after its first document is stored.
- Uploads are never written to disk. Only extracted text and embeddings stay
  in memory.
- CPU and LLM work runs in worker threads. Semaphores allow 4 concurrent LLM
  calls and 2 ingestions; a request that waits more than 20s gets a 503
  "busy" response.

## Retrieval

- **Hybrid.** Each query ranks all chunks of the selected documents by BM25
  (statistics over those documents together) and by MiniLM cosine. The
  rankings are fused with RRF and the top 4 are kept. A follow-up adds the
  query "previous question + question". The displayed `score` is cosine. With
  one document and one query this equals the "Hybrid" retriever in
  `retrieval_eval.py` (Hit@4 0.82), and a test pins that.
- **Overview routing.** `is_overview_question()` routes whole-document
  questions ("what is this about", "summarize", "main focus / purpose of this
  resume") to `overview_sample()`. "Summarize section 4" names a specific part,
  so it uses normal retrieval.
- **Optional floor.** When `RETRIEVAL_SCORE_FLOOR > 0` and every hit scores
  below it, no passages are sent and the model refuses. The default is off.
  Similarity alone can't tell short or pronoun-heavy questions from
  off-topic ones (`reports/score_floor_eval.md`).

## Generation

- **System prompt:** answer directly and then explain; use only the passages;
  match the requested level; cite `[Page N]` / `[file, Page N]` after claims;
  answer what is covered and name what isn't; otherwise reply with exactly
  "I could not find the answer to this question in the document."
- **Passages are untrusted.** They are fenced between `<<<BEGIN PASSAGES>>>`
  and `<<<END PASSAGES>>>`. Fence tokens inside passage text, filenames and
  earlier turns are replaced with look-alike characters. A reminder after the
  question repeats the rule, and the summary and suggestion prompts carry it
  too. Earlier turns are labelled as untrusted references.
- **Providers.** `providers.py` builds a chain from every configured key
  (Groq, OpenRouter, NVIDIA, Hugging Face; Google opt-in) in `LLM_PROVIDERS`
  order, plus `LLM_FALLBACK_MODEL`. A route that fails (429, 5xx, 401/403/404,
  timeout, empty reply, dropped stream) passes the question on and cools down:
  as `Retry-After` says (≤15 min), 15 min for a bad key, or 30s for a
  transient error. Streams switch only before the first token. Every answer
  carries its route (`answered_by` / `route` event).
- **Streaming.** `/api/ask/stream` resolves sources and the first piece before
  responding, so setup errors still return proper status codes. Events:
  `sources`, `reasoning` (display-only, never stored, and discarded if its
  route fails), `route`, `token`, `done`/`error`. A turn is saved only when
  the stream completes. `X-Accel-Buffering: no` keeps Nginx from buffering.
- **Summary** uses 10 evenly spaced chunks. **Suggestions** come from the
  overview sample.

## Frontend

- Vanilla JS, with all dynamic text inserted via `textContent` (never
  `innerHTML`). A small safe Markdown renderer handles answers.
- Page references in answers become `p. N` chips that highlight the matching
  source card. Source cards sit beside answers on wide screens (container
  query) and fold under them on phones. On a refused answer they are dimmed
  and labelled as passages searched.
- Streamed answers are announced to screen readers through a live region.
  Controls are keyboard-reachable with visible focus.
- One `api()` helper turns network and proxy errors into readable messages.
  Asset URLs carry `?v=<APP_VERSION>`, and a test checks they match
  `main.py`.

## Security controls

| Threat | Control |
|---|---|
| Quota/CPU abuse | Per-IP limits: LLM 10/min and 100/h, uploads 10 per 10 min. `X-Real-IP` is trusted only from `TRUSTED_PROXIES`. |
| Memory exhaustion | 25MB uploads (never read past), 1500 chunks/doc (~360 pages, checked before embedding), `MAX_TOTAL_CHUNKS`, 5 docs, 50 sessions |
| Bad uploads | `%PDF-` signature for PDFs; filename reduced to basename, control characters stripped, 120 characters |
| XSS / clickjacking | CSP `'self'` with no inline code, `frame-ancestors 'none'`, `X-Frame-Options: DENY`, `nosniff`, `textContent` only |
| CSRF | SameSite=Lax plus a check that refuses cross-origin or cross-site API POSTs |
| Cookie tampering | `__Host-` prefix over HTTPS |
| Recon / error leaks | API docs disabled; generic errors with a reference code |
| Large JSON | 16KB cap |
| IDOR | Document ids are looked up only within the caller's session |
| Prompt injection | Fenced untrusted passages, neutralised fence tokens, reminder after the question |
| Dependencies | Dependabot, `pip-audit` and CodeQL in CI |

## Failure handling

| Situation | Result |
|---|---|
| Unsupported type, or 6th document | 400 |
| Over 25MB / too much text | 413 |
| No extractable text | 422 |
| Server chunk budget full | 503 |
| No provider configured | 503 |
| All providers failing | 502 (an `error` event once streaming has started) |
| Anything unexpected | 500 with a reference code; details only in the log |
| Not answerable | Exact refusal sentence |

## Deployment

A single EC2 `t3.small` runs the container behind Nginx (Let's Encrypt,
`client_max_body_size 25m`, 120s timeouts). The container is bound to
`127.0.0.1:8000`, with `--restart unless-stopped` and log rotation
(`max-size=10m`, `max-file=3`). Docker's port mapping makes Nginx appear as
`172.17.0.1`, so `TRUSTED_PROXIES` includes it. Releases build a thin layer
over the base image (`deploy/Dockerfile.update`). Sessions live in memory, so
the design assumes one instance.
