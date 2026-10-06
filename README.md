# DocuLens — An AI Powered Document Assistant

A Retrieval-Augmented Generation (RAG) app that answers questions about the
documents you upload. Answers come only from the documents, and every answer
shows the exact page and passage it is based on. If the documents don't
contain the answer, it says so instead of guessing.

**Live:** https://doculens.duckdns.org — **Latest release:** v4.0.0-beta.2 (pre-release)

## Key features

- **Any document:** PDF, Word (.docx), text and Markdown. Scanned pages are
  OCR'd page by page (Tesseract), mixed documents included. Up to 5 documents,
  searched together or individually.
- **Saved documents (sign in):** with Firebase sign-in, a user's documents are
  stored (compressed text + embeddings, never the original file) and survive
  restarts and other instances; guests get 1 document and 5 questions a day,
  held in memory only.
- **Verifiable answers:** source passages appear beside each answer, and page
  references become clickable `p. 3` chips that highlight the passage. When the
  model can't answer, the cards are dimmed and labelled as passages searched,
  not evidence.
- **Grounded:** the model answers only from the retrieved passages, or returns
  one exact refusal sentence. Document text is treated as untrusted data,
  never as instructions.
- **Hybrid retrieval:** BM25 fused with MiniLM embeddings (reciprocal rank
  fusion) over structure-aware chunks (sentences, tables and headings kept
  intact, each chunk knows its section). Exact vector search by default, HNSW
  above `ANN_THRESHOLD`. Whole-document questions ("what is the main focus of
  this resume?") use an ordered sample of the document instead.
- **Conversation:** streamed answers with Stop, follow-ups resolved from the
  earlier questions ("Who founded it?"), tailored "Try asking" starters, copy
  and Markdown export, and session restore after a reload.
- **Real summaries:** a document is read in batches and merged (map-reduce)
  within an LLM-call budget, with page ranges kept; if it is too long for the
  budget, the summary says which parts it did not read.
- **Shows the model's thinking** in a collapsible panel when the provider
  exposes it. Reasoning is display-only and never stored.
- **Any model, with failover:** OpenAI-compatible services (Groq, OpenRouter,
  NVIDIA Nemotron/gpt-oss, Hugging Face, OpenAI, Mistral, DeepSeek, xAI,
  Together, Azure, Ollama/vLLM), Anthropic, Gemini and Cohere, in any order, via
  presets, `<NAME>_MODEL` lists or `LLM_ROUTES`. A failing or rate-limited
  provider cools down and the next one answers. Each answer names the model that
  wrote it.
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
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `COHERE_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `TOGETHER_API_KEY` | empty | More providers; add the name to `LLM_PROVIDERS` |
| `LLM_PROVIDERS` | `groq,openrouter,nvidia,huggingface` | Failover order |
| `<NAME>_MODEL`, `<NAME>_BASE_URL` | per provider | Override a provider's model or endpoint; `<NAME>_MODEL` may list several models (comma-separated), each becomes a route |
| `LLM_ROUTES` | empty | JSON list for any other endpoint or model family (`api`: openai/anthropic/gemini/cohere); see `providers.py` |
| `CHUNKER` | `structured` | `char` restores the fixed character window |
| `VECTOR_INDEX`, `ANN_THRESHOLD` | `auto`, `20000` | `flat`/`hnsw`/`auto`; vectors at which `auto` switches to HNSW |
| `REWRITE_LLM` | `auto` | Follow-up rewriting: `off`, `auto` (hard cases only), `always` |
| `SUMMARY_MAX_LLM_CALLS`, `SUMMARY_BATCH_CHARS`, `SUMMARY_CONCURRENCY` | `10`, `6000`, `3` | Summary budget and batch size |
| `OCR_MAX_PAGES`, `OCR_TIME_BUDGET`, `OCR_LANG`, `PDF_MAX_PAGES` | `30`, `60` s, `eng`, `2000` | OCR and extraction limits |
| `DOC_RETENTION_DAYS`, `USER_STORAGE_MB`, `ARTIFACT_DIR` | `30`, `20`, empty | Saved-document retention, per-user storage allowance, optional directory instead of Redis for the blobs |
| `LLM_FALLBACK_MODEL` | empty | A second Groq model tried last |
| `TRUSTED_PROXIES` | `127.0.0.0/8,::1/128` | CIDRs whose `X-Real-IP` is trusted for rate limiting. Behind Docker's port mapping add the bridge gateway (`172.17.0.1/32`). Never `0.0.0.0/0`. |
| `FIREBASE_PROJECT_ID`, `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN` | empty | Firebase Authentication (Google provider). Public web config, no server secret. Unset = sign-in off and no limits (local runs, tests). Add the site's domain under Firebase's Authorized domains |
| `GUEST_MAX_DOCS`, `GUEST_DAILY_QUESTIONS`, `GUEST_DAILY_UPLOADS` | `1`, `5`, `3` | Guest limits (per IP, sliding 24 h); signed-in users get `USER_DAILY_QUESTIONS` / `USER_DAILY_UPLOADS` (`200`, `30`) and 5 documents |
| `REDIS_URL` | empty | Shared store for logins and rate limits, e.g. [Upstash](https://upstash.com) (`rediss://default:<password>@<name>.upstash.io:6379`; pick the Mumbai region next to the server). Unset = in-process. If Redis fails the app keeps serving with in-process state |
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

The original file is never stored. A guest's extracted text lives in server
memory and is deleted on removal, after 2 hours idle, or on restart. A signed-in
user's extracted text and embeddings are saved (compressed) in Redis for 30 days
(`DOC_RETENTION_DAYS`) and deleted when the document is removed.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /api/ingest` | Upload a document (multipart) |
| `POST /api/ask/stream` | Server-sent events: `sources`, optional `reasoning`, `route`, `token`…, `done` / `error` |
| `POST /api/ask` | Same answer as one JSON response |
| `POST /api/summary`, `POST /api/suggestions` | Summary or starter questions for one document |
| `POST /api/remove` | Remove one document, or everything |
| `GET /api/session` | Documents and conversation, for restoring after a reload |
| `POST /api/login` | Exchange a Firebase ID token for a login session |
| `GET /api/me`, `POST /api/logout` | Current user, limits and public Firebase config; sign out (also drops loaded documents) |
| `GET /api/status`, `GET /api/health` | Version and provider status; health check |

`/api/ask*` accept an optional `doc_ids` list to limit the search.

## Architecture

FastAPI serves both the API and a static HTML/CSS/JS frontend from one
origin. The pipeline is plain Python with no LangChain:

```
upload → extract pages (PyMuPDF / per-page OCR / docx) → structure-aware chunks (<= 800 chars)
       → embed (all-MiniLM-L6-v2) → vector index (exact or HNSW) + BM25 statistics
       → signed-in: chunks + embeddings saved (Redis); the index is rebuilt on load
question → follow-up resolved from earlier questions → BM25 + embedding search, fused
         → top 4 passages → prompt (rules + fenced passages + last 3 turns) → LLM → answer + sources
summary  → all chunks → batches → batch summaries → merged → final (<= 10 LLM calls)
```

| File | Role |
|---|---|
| `document_loader.py`, `pdf_loader.py` | Text extraction by file type, per-page OCR |
| `chunker.py`, `embedder.py`, `vector_index.py`, `vector_store.py`, `retriever.py` | Chunking, embeddings (cached), exact/HNSW index, BM25 + fusion |
| `conversation.py`, `summarizer.py` | Follow-up resolution, hierarchical summaries |
| `llm_client.py`, `llm_adapters.py`, `providers.py` | Prompts, streaming, API families, provider chain and failover |
| `auth.py`, `store.py`, `docstore.py` | Firebase token verification, Redis/in-process state, persisted documents |
| `pipeline.py` | Ingest, retrieve, answer, summarize |
| `main.py` | API, sessions, limits, security middleware |
| `static/` | Frontend |
| `evaluate.py`, `retrieval_eval.py`, `conversation_eval.py`, `bench_vector_index.py`, `bench_latency.py`, `score_floor_eval.py` | Evaluation and benchmarks |

Details: [`ARCHITECTURE.md`](ARCHITECTURE.md). Design decisions and their
history: [`DECISIONS.md`](DECISIONS.md).

## Security

- **Prompt injection:** passages are fenced and declared untrusted; fence
  tokens inside uploaded text and filenames are neutralised; earlier turns are
  labelled untrusted. Tested in `tests/test_prompt_injection.py`.
- **Headers:** strict CSP (`'self'` only, no inline code), HSTS, `nosniff`,
  `X-Frame-Options: DENY`, no-referrer. API docs endpoints are disabled.
- **Sessions:** `__Host-` cookies (HttpOnly, SameSite=Lax, Secure over HTTPS);
  documents sit under a sliding 2-hour session, logins under a 7-day one;
  cross-site API requests are refused.
- **Sign-in:** Firebase Authentication (Google). The browser gets an ID token;
  the server verifies its RS256 signature against Google's published
  certificates (audience, issuer, expiry, verified e-mail), then issues its own
  login cookie. The SDK is served from this origin; CSP gains only the Google
  hosts it needs and COOP becomes `same-origin-allow-popups`, only when sign-in
  is configured.
- **Tiers:** guests get 1 document, 5 questions and 3 uploads per day per IP;
  signed-in users get 5 documents and far higher caps, counted per account.
- **Limits:** per-IP rate limits (10/min and 100/h for LLM calls, 10 uploads
  per 10 min), 25MB uploads, ~360 pages per document, 5 documents per session,
  50 sessions, `MAX_TOTAL_CHUNKS`, 16KB JSON bodies, 4 concurrent LLM calls and
  2 ingestions.
- **Data:** keys come only from the environment and are never logged or shown;
  only the question and retrieved passages are sent to the AI provider; the
  page loads nothing from other sites.
- **CI:** CodeQL and `pip-audit` on every push; Dependabot updates.
- **Uptime:** a scheduled GitHub Actions workflow (`uptime.yml`) checks the live health, status and front page every 15 minutes; a failure emails the repository owner.

Report vulnerabilities as described in [`SECURITY.md`](SECURITY.md).

## Evaluation

**Tests.** `pip install -r requirements-dev.txt && pytest` runs 480 tests; the
few that call a real LLM skip without a key. CI also runs headless-browser
end-to-end tests (upload, ask, citations, sign-in states, summaries, warnings).

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

**v4 measurements** (800/150, Hybrid, MiniLM; small sets, one document, so
read differences of a question or two as noise):

| Set | Chunker | Hit@1 | Hit@4 | MRR@10 | Page-Hit@4 |
|---|---|---|---|---|---|
| v1, 39 questions (wording shared with the text) | char | 0.56 | 0.82 | 0.70 | 0.82 |
| | **structured** | 0.59 | 0.82 | 0.70 | 0.82 |
| v2, 37 harder questions (paraphrase, numbers, tables, short, cross-page) | char | 0.51 | 0.73 | 0.62 | 0.78 |
| | **structured** | 0.62 | 0.78 | 0.70 | 0.81 |

Follow-ups (`conversation_eval.py`, 15 questions): searched alone Hit@4 0.67;
v3 concatenation 0.87; v4 resolver 0.93 (MRR 0.68 → 0.73, Page-Hit@4 0.93 →
0.87). HNSW vs exact on the same questions gives identical results; on
synthetic vectors (`bench_vector_index.py`) HNSW is 10-20x faster from 20k
vectors with recall@10 of 1.0 on clustered data but only 0.3-0.7 on random
data, so exact search stays the default (a single document is at most 1500
chunks). Stage latencies are in [`reports/latency_bench.md`](reports/latency_bench.md).

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

- Signed-in documents persist (Redis) and load lazily into a bounded in-memory
  cache; guests' documents are memory-only. The original upload is not kept, so
  a document whose stored data is lost must be uploaded again.
- Up to 5 documents per user and 20 MB of saved data; not built for large
  corpora. The chunk budget is a soft limit.
- Follow-ups are resolved from the earlier *questions* only (never answers);
  references to something only an earlier answer said are not resolved.
- Summaries read the whole document only if it fits the call budget (about
  100k characters by default); longer ones report what was skipped.
- Embedding is CPU-bound: about 6-25 ms per chunk on a fast machine depending
  on how token-dense the text is, several times that on the 2-vCPU server.
- OCR reads printed text in the configured language(s) (English by default), up to 30 scanned pages and 60 s per document; handwriting is not supported.
- One EC2 instance with no external monitoring or backups.

## Future improvements

Ingestion progress for large files, cross-encoder reranking, a hosted-embedding option for slow hosts, and a CLI/setup wizard.
