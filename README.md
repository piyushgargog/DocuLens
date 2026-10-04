# DocuLens — An AI Powered Document Assistant

A Retrieval-Augmented Generation (RAG) app that answers questions about the
documents you upload. Answers come only from the documents, and every answer
shows the exact page and passage it is based on. If the documents don't
contain the answer, it says so instead of guessing.

**Live:** https://doculens.duckdns.org — **Latest release:** v3.7.0

## Key features

- **Any document:** PDF, Word (.docx), text and Markdown. Scanned PDFs are
  OCR'd (Tesseract). Up to 5 documents per session, searched together or
  individually.
- **Verifiable answers:** source passages appear beside each answer, and page
  references become clickable `p. 3` chips that highlight the passage. When the
  model can't answer, the cards are dimmed and labelled as passages searched,
  not evidence.
- **Grounded:** the model answers only from the retrieved passages, or returns
  one exact refusal sentence. Document text is treated as untrusted data,
  never as instructions.
- **Hybrid retrieval:** BM25 fused with MiniLM embeddings (reciprocal rank
  fusion). Whole-document questions ("summarize this", "what is the main focus
  of this resume?") use an ordered sample of the document instead.
- **Conversation:** streamed answers with Stop, follow-up questions, one-click
  summaries, tailored "Try asking" starters, copy and Markdown export, and
  session restore after a reload.
- **Shows the model's thinking** in a collapsible panel when the provider
  exposes it. Reasoning is display-only and never stored.
- **Provider failover:** Groq, OpenRouter, NVIDIA and Hugging Face (Gemini
  opt-in). A failing or rate-limited provider cools down and the next one
  answers. Each answer names the model that wrote it.
- **Accessible, framework-free UI:** "Calm Light" design, light and dark
  themes, keyboard navigation, screen-reader announcements for streamed
  answers, mobile layout. See [`DESIGN.md`](DESIGN.md).

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/piyushgargog/DocuLens.git
cd DocuLens
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add at least one provider key
uvicorn main:app --reload
```

Open http://localhost:8000.

## Configuration

Set at least one provider key in `.env`. Each extra key adds a fallback.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `HF_TOKEN` | empty | Provider keys (`LLM_API_KEY` still works as the Groq key) |
| `LLM_PROVIDERS` | `groq,openrouter,nvidia,huggingface` | Failover order; add `google` + `GEMINI_API_KEY` for Gemini |
| `<NAME>_MODEL`, `<NAME>_BASE_URL` | per provider | Override a provider's model or endpoint |
| `LLM_FALLBACK_MODEL` | empty | A second Groq model tried last |
| `TRUSTED_PROXIES` | `127.0.0.0/8,::1/128` | CIDRs whose `X-Real-IP` is trusted for rate limiting. Behind Docker's port mapping add the bridge gateway (`172.17.0.1/32`). Never `0.0.0.0/0`. |
| `MAX_TOTAL_CHUNKS` | `75000` | Server-wide index budget (~3 KB per chunk); 503 when full |
| `PREFETCH_MODEL` | `0` | `1` loads the embedding model at startup |
| `RETRIEVAL_SCORE_FLOOR` | `0` (off) | Opt-in similarity floor; see [evaluation](#evaluation) before enabling |
| `LOG_LEVEL`, `SENTRY_DSN` | `INFO`, empty | Logging; opt-in error reporting (no document content is sent) |

## Usage

1. Upload a document (25MB limit) by clicking or dropping it anywhere.
2. Ask a question; press Enter (Shift+Enter for a new line, `Esc` to stop).
3. Check the source cards and click `p. N` chips to see the exact passage.
4. Ask follow-ups, use **Explain more simply** / **Go deeper**, or
   **Summarize** a document.
5. Remove a document with **×**, or everything with **Remove all**.

Documents are never written to disk. Extracted text lives in server memory
and is deleted on removal, after 2 hours idle, or on restart.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /api/ingest` | Upload a document (multipart) |
| `POST /api/ask/stream` | Server-sent events: `sources`, optional `reasoning`, `route`, `token`…, `done` / `error` |
| `POST /api/ask` | Same answer as one JSON response |
| `POST /api/summary`, `POST /api/suggestions` | Summary or starter questions for one document |
| `POST /api/remove` | Remove one document, or everything |
| `GET /api/session` | Documents and conversation, for restoring after a reload |
| `GET /api/status`, `GET /api/health` | Version and provider status; health check |

`/api/ask*` accept an optional `doc_ids` list to limit the search.

## Architecture

FastAPI serves both the API and a static HTML/CSS/JS frontend from one
origin. The pipeline is plain Python with no LangChain:

```
upload → extract pages (PyMuPDF / OCR / docx) → chunk (800/150 chars)
       → embed (all-MiniLM-L6-v2) → FAISS index per document (in memory)
question → BM25 + embedding search, fused → top 4 passages
         → prompt (rules + fenced passages + last 3 turns) → LLM → answer + sources
```

| File | Role |
|---|---|
| `document_loader.py`, `pdf_loader.py` | Text extraction by file type, OCR fallback |
| `chunker.py`, `embedder.py`, `vector_store.py`, `retriever.py` | Chunking, embeddings, FAISS, BM25 + fusion |
| `llm_client.py`, `providers.py` | Prompts, streaming, provider failover |
| `pipeline.py` | Ingest, retrieve, answer, summarize |
| `main.py` | API, sessions, limits, security middleware |
| `static/` | Frontend |
| `evaluate.py`, `retrieval_eval.py`, `score_floor_eval.py` | Evaluation scripts |

Details: [`ARCHITECTURE.md`](ARCHITECTURE.md). Design decisions and their
history: [`DECISIONS.md`](DECISIONS.md).

## Security

- **Prompt injection:** passages are fenced and declared untrusted; fence
  tokens inside uploaded text and filenames are neutralised; earlier turns are
  labelled untrusted. Tested in `tests/test_prompt_injection.py`.
- **Headers:** strict CSP (`'self'` only, no inline code), HSTS, `nosniff`,
  `X-Frame-Options: DENY`, no-referrer. API docs endpoints are disabled.
- **Sessions:** anonymous `__Host-` cookie (HttpOnly, SameSite=Lax, Secure over
  HTTPS) with a sliding 2-hour expiry; cross-site API requests are refused.
- **Limits:** per-IP rate limits (10/min and 100/h for LLM calls, 10 uploads
  per 10 min), 25MB uploads, ~360 pages per document, 5 documents per session,
  50 sessions, `MAX_TOTAL_CHUNKS`, 16KB JSON bodies, 4 concurrent LLM calls and
  2 ingestions.
- **Data:** keys come only from the environment and are never logged or shown;
  only the question and retrieved passages are sent to the AI provider; the
  page loads nothing from other sites.
- **CI:** CodeQL and `pip-audit` on every push; Dependabot updates.

Report vulnerabilities as described in [`SECURITY.md`](SECURITY.md).

## Evaluation

**Tests.** `pip install -r requirements-dev.txt && pytest` runs 150+ tests;
the few that call a real LLM skip without a key. CI also runs a headless-browser
end-to-end test (upload, ask, citations, refusal, keyboard access).

**Answer quality** (`evaluate.py`, "Attention Is All You Need", 6 questions):
small chunks (300/50, top 3) wrongly refused 4 of 5 answerable questions;
large chunks (1000/200, top 5) answered all 5 with correct pages. Both refused
the unanswerable one. The default (800/150, top 4) follows from this.

**Retrieval** (`retrieval_eval.py`, 39 labelled questions, no LLM calls) at
800/150:

| Retriever | Hit@1 | Hit@4 | MRR@10 |
|---|---|---|---|
| BM25 | 0.64 | **0.87** | **0.75** |
| MiniLM | 0.38 | 0.77 | 0.56 |
| BGE-small | 0.59 | 0.77 | 0.68 |
| **Hybrid (used)** | 0.56 | 0.82 | 0.70 |

300-character chunks were worst for every retriever. The questions share the
paper's wording, which favours BM25. Full results:
[`reports/retrieval_eval.md`](reports/retrieval_eval.md).

**Similarity floor** (`score_floor_eval.py`): on the sample documents a 0.25
floor looked safe, but on a real one-page resume answerable questions scored
0.06–0.27, the same range as off-topic ones. The floor is therefore off by
default. See [`reports/score_floor_eval.md`](reports/score_floor_eval.md).

## Deployment

Live on an AWS EC2 `t3.small` (2 vCPU, 2GiB) in `ap-south-1`: Docker
(`--restart unless-stopped`, logs capped with `--log-opt max-size=10m
--log-opt max-file=3`) behind Nginx with a Let's Encrypt certificate. The app
binds to `127.0.0.1`; Nginx sets `X-Real-IP`, and `TRUSTED_PROXIES` includes
the Docker bridge gateway. Releases are built as a thin layer over the base
image (`deploy/Dockerfile.update`) because the disk is small.

To run anywhere:

```bash
docker build -t doculens .
docker run -p 8000:8000 --env-file .env doculens
```

The `Dockerfile` installs CPU-only PyTorch to avoid several hundred MB of
unused CUDA packages.

## Known limitations

- Sessions live in process memory: lost on restart and not shared across
  replicas. Scaling out needs a shared store such as Redis.
- Up to 5 documents per session; not built for large corpora. The chunk budget
  is a soft limit.
- Follow-ups use the last 3 turns and combine the previous question with the
  new one (no query rewriting).
- Summaries use 10 evenly spaced excerpts, not the whole document.
- Character-based chunking can split a sentence or table row.
- OCR is English-only and limited to 30 pages.
- One EC2 instance with no external monitoring or backups.

## Future improvements

Ingestion progress for large files, follow-up query rewriting, map-reduce
summaries, section-aware chunking, cross-encoder reranking, and a shared
session store.
