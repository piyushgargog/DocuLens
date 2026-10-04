# Architecture — DocuLens

_Current as of v3.6.1._

## Overview

A small FastAPI backend implementing a Retrieval-Augmented Generation (RAG)
pipeline from explicit, individually-inspectable steps rather than a
framework's black-box chain (see `DECISIONS.md` decision #1). Each pipeline
stage is a plain function you can point to and explain. The same process
serves a static HTML/CSS/JS frontend, so API and page share one origin and
no CORS configuration exists anywhere.

The UI was originally Streamlit, replaced by FastAPI + a static frontend
for mobile usability; the frontend has since been redesigned twice (v1.3.0,
v1.4.0 — see `DESIGN.md`). None of those changes touched the retrieval or
prompting code, and the single-document path is guarded by tests so the
documented chunking evaluation stays valid.

## Components

1. **Backend / API** (`main.py`, FastAPI)

   | Endpoint | Does |
   |---|---|
   | `POST /api/ingest` | Upload a PDF (≤25MB). Adds it to the session (creating one and setting the cookie if needed), max 5 documents. |
   | `POST /api/ask/stream` | `{question, doc_ids?}` → server-sent events: `sources`, then `token` pieces as the model writes, then `done` (or `error`). Used by the UI. |
   | `POST /api/ask` | Same request → one JSON `{answer, sources}`. |
   | `POST /api/summary` | `{id}` → a short summary of one document. |
   | `POST /api/suggestions` | `{id}` → up to 4 starter questions the LLM writes from a sample of the document. |
   | `POST /api/remove` | `{id}` removes one document; no id (or removing the last one) clears the session and cookie. |
   | `GET /api/session` | The session's documents and conversation, so a page reload restores the UI. |
   | `GET /`, `/static/*` | The frontend, sent with `Cache-Control: no-cache` so browsers never run a stale script after a deploy. |

   State is an in-memory `dict[str, Session]` keyed by an `httponly`,
   `samesite=lax` cookie (`Secure` when the request arrived over HTTPS,
   directly or via Nginx's `X-Forwarded-Proto`). A `Session` holds up to 5
   documents (each its own `IndexState`) and the last 10 Q/A turns. Sessions
   expire after 2 hours idle — enforced by a background task (started in the
   app's `lifespan`) every 5 minutes, not only when a request arrives — and at
   most 50 are kept (oldest-idle evicted). A configurable `MAX_TOTAL_CHUNKS`
   budget (75,000 by default) provides an additional aggregate memory guard.
   Active API responses refresh the browser cookie expiry without rotating the
   anonymous session ID. Each `Session` has its own `RLock` guarding its
   documents, history and last-used time, because ingestion, answering and the
   sweeper run on different worker threads. Uploaded documents
   are never written to disk: the upload's temporary spool file is closed
   right after reading, and only extracted text and embeddings are kept, in
   memory, for the life of the session. Ingestion and LLM calls run in a worker thread
   (`run_in_threadpool`) so one slow request never blocks the event loop.
   Errors use real status codes (400/404/413/422/502/503/500) with a fixed,
   generic message; exception text is logged server-side only.

2. **Frontend** (`static/index.html`, `static/style.css`, `static/app.js`)
   Vanilla, no framework or build step. The design ("Calm Light", v3.4.0 —
   modelled on Claude's own design language: ink text and black primary
   buttons on a warm near-white ground, hairline borders, one terracotta
   accent for brand and citation chips, self-hosted Newsreader serif for
   the hero and wordmark, Inter for the UI; see `DESIGN.md`): a documents sidebar card and a conversation card;
   answers carry their retrieved passages as source cards beside them on
   wide screens (a CSS container
   query on the conversation pane) and folded under them on narrow ones.
   Page references the model writes (`【file, Page 3】`, `(Page 3)`,
   `[Page 3]`, …) are parsed into "p. 3" chips; clicking one
   highlights the matching passage in yellow. Model answers render as safe Markdown (v3.2.0:
   headings, lists, code, tables, bold, inline citation chips) through a
   small renderer in `app.js`. Every dynamic string is inserted with
   `textContent` or text nodes, never `innerHTML`. All requests go through
   one `api()` helper that turns network failures and non-JSON proxy error
   pages into readable messages instead of a stuck UI. The release version
   lives once in `main.py` (`APP_VERSION`, also the FastAPI app version);
   the page repeats it in its asset URLs and footer, and a test fails if
   they drift apart.

3. **Document loader** (`pdf_loader.py`)
   PyMuPDF extracts text page by page into `[(page_number, text), ...]`,
   skipping empty pages. A PDF with no text layer anywhere (a scan) is
   OCR'd instead (Tesseract via pytesseract, up to 30 pages, ~144 DPI). An
   invalid, encrypted or unreadable PDF yields an
   empty list rather than an exception.

4. **Chunker** (`chunker.py`)
   A character sliding window per page (`chunk_size`, `chunk_overlap`).
   Every chunk keeps its page number: `{"text", "page"}`.

5. **Embedder** (`embedder.py`)
   `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized 384-dim vectors.
   The model is a module-level singleton (loaded once per process), plain
   Python so it works the same from `main.py`, `evaluate.py` and tests. With
   `PREFETCH_MODEL=1` the app's `lifespan` loads it at startup
   (`embedder.prefetch()`, in a worker thread) so the first upload doesn't pay
   the ~2 s load; otherwise it loads lazily on first use.

6. **Vector store** (`vector_store.py`)
   One FAISS `IndexFlatIP` per document; on normalized vectors inner product
   equals cosine similarity. `search()` returns `{text, page, score}` plus
   any extra chunk keys (e.g. `doc`).

7. **LLM client** (`llm_client.py`)
   OpenAI-compatible chat-completions calls over `requests`. Builds the
   grounding prompt (below) and sends it along the provider chain from
   `providers.py` (v2.2.0): Groq, OpenRouter, NVIDIA and Hugging Face
   (Google AI Studio opt-in), in
   `LLM_PROVIDERS` order, each used only if its key is set, plus an optional
   second Groq model (`LLM_FALLBACK_MODEL`) at the end. A route that is rate
   limited, times out, returns 5xx/401/403/404 or an empty reply hands the
   question to the next and goes on cooldown (as long as `Retry-After` asks,
   capped at 15 min; 15 min for a bad key or model; 30s for a timeout or
   5xx). While another route remains, a 429 is not waited out (at most 3s);
   the last route waits up to 30s as before. Streams switch provider only
   before the first token. Replies carry their route (`Answer`, a `str`
   subclass), so the API can say who answered. Failures surface as
   `LLMConfigError` (no provider configured) / `LLMRateLimitError` (all
   rate limited) / `LLMRequestError`.

8. **Orchestration** (`pipeline.py`)
   `ingest()`, `retrieve()`, `answer()` and `summarize()` — the only place
   the stages are wired together.

## Data Flow

```
PDF upload ──► POST /api/ingest
                 │ pdf_loader: [(page, text), ...]
                 │ chunker:    [{text, page, doc}, ...]
                 │ embedder:   384-dim vectors
                 ▼
               IndexState (FAISS index + chunks) ──► session.docs[doc_id]

question ──► POST /api/ask
               │ queries = [question] (+ "previous question + question" for a follow-up)
               │ for each document × each query: FAISS top-k
               │ merge, keep best score per passage, overall top-k
               ▼
             prompt = system rules + fenced passages "[file.pdf, Page N] ..."
                      + last 3 turns (if any) + question
               │ LLM API
               ▼
             {answer, sources} ──► app.js: answer text with page tabs,
                                   sources as margin notes
```

## RAG Pipeline (detail)

**Ingestion (once per uploaded PDF).** Extract per page, chunk with overlap
so an answer spanning a boundary isn't lost, tag each chunk with the
document's filename, embed, and build that document's FAISS index. Two
uploads with the same filename are named `a.pdf` and `a.pdf (2)` so sources
stay unambiguous.

**Retrieval (per question) — hybrid since v2.0.0.** All chunks of the
selected documents are ranked twice per query: by BM25 (`retriever.py`;
term statistics taken over the selected documents together, so scores are
comparable across them) and by cosine similarity of the MiniLM embeddings.
Every ranking is fused with reciprocal rank fusion (k = 60) and the top 4
kept (`DEFAULT_TOP_K`). For a follow-up there are two queries — the question
and "previous question + question" — so four rankings are fused. Each
passage keeps its cosine similarity as the displayed `score`. With one
document and one query this is exactly the "Hybrid" retriever measured in
`retrieval_eval.py` (Hit@4 0.82, MRR@10 0.70 vs 0.77 / 0.56 for embeddings
alone); a test pins that equivalence, and `retrieval_eval.py` imports the
same functions.

**Optional abstention floor (v3.6.0; off by default since v3.6.1).** When
`RETRIEVAL_SCORE_FLOOR` is above 0, `gather_sources()` compares the best
displayed cosine score of the top-k hits of a non-overview question with it; if
even the best is below, no passages are passed on (the prompt contains
"(no passages retrieved)") and the model returns its exact refusal. It shipped on
at 0.25 after `score_floor_eval.py` looked safe on two sample documents, then
refused real questions on a one-page resume (answerable questions scored
0.06-0.27, an unrelated one 0.05): short documents and generic or pronoun-heavy
questions have low cosine similarity to their own answers, so similarity alone
cannot separate them from off-topic questions. The prompt's refusal rule remains
the hallucination control; the floor stays as an opt-in knob, and
`reports/score_floor_eval.md` documents the measurement and its limits. Overview questions bypass it (they use an ordered sample, not
similarity); "summarize section 4" is recognised as a specific-part question
(`_SECTION_QUALIFIER`) and goes through retrieval and the floor.

**Routing whole-document questions.** Similarity search answers "where
does the document talk about X". A question about the document as a whole
("what is this paper about?", "main contribution", "summarize the key
findings") resembles no passage — in testing it retrieved the reference
list, and the assistant refused. `pipeline.is_overview_question()` (a
regular expression, tested against both kinds of question) routes these to
`overview_sample()`: the document's first two chunks (abstract/introduction)
plus evenly spaced chunks, eight passages in total, split across documents.

**Generation (v1.7.0 prompt).** The LLM's job is to answer, not to quote. The
system prompt asks for a direct answer first, then explanation in its own
words, combining passages where that helps; to match the level the user asks
for; plain text with `- ` bullets (no LaTeX, tables or headings); a page
citation after each claim; and, when the passages cover only part of the
question, to answer that part and say what's missing. The grounding
contract is unchanged: no facts, numbers or examples beyond the passages,
one exact refusal string when nothing is relevant, and passages declared
untrusted content fenced between `<<<BEGIN PASSAGES>>>` / `<<<END
PASSAGES>>>`, so an instruction inside a PDF is reported, not obeyed
(`tests/test_prompt_injection.py`). Delimiter tokens found inside uploaded
passage text and filenames are replaced with inert Unicode lookalikes before
prompt construction. Earlier turns may resolve references but are never
trusted as evidence. Whole-document overview questions use a bounded sample;
section-specific requests stay on normal retrieval and abstention paths.

**Streaming.** `llm_client.ask_stream()` sends `stream: true` and yields
content deltas from the provider's SSE stream (decoded as UTF-8 explicitly —
providers omit the charset and `requests` would otherwise assume
ISO-8859-1). Providers that expose their reasoning (Groq's gpt-oss, OpenRouter's
Nemotron) stream it as a separate delta; it is forwarded as a `reasoning` SSE
event for the collapsible "Thinking" panel, but is display-only — never saved
to the history, never sent back to the model, and a reasoning stream from a
route that then fails is discarded. `main.py` fetches the sources and the
*first* piece before opening the response, so configuration, rate-limit and
provider errors still return a normal HTTP status; after that, a failure
becomes an `error` event. A connection that drops mid-stream is wrapped in
`LLMRequestError` so provider failover and cooldown still apply.
The turn is saved to the history only when the stream completes, so a
stopped answer never becomes context. The response carries
`X-Accel-Buffering: no` so Nginx passes events through immediately.

**Provider failover (v2.2.0).** Free tiers are small — Groq's is 200k
tokens a day per model — so one provider alone runs out. `providers.py`
builds a chain of every provider with a key (Groq, OpenRouter, NVIDIA and
Hugging Face serving open models: gpt-oss-120b, Nemotron 3 Super,
gpt-oss-20b; Google AI Studio's Gemini is opt-in), then Groq's
`LLM_FALLBACK_MODEL`. A failing route is skipped for a cooldown so later
questions don't wait on it; if every route is cooling down they are all
tried anyway. The `route` SSE event and `answered_by` fields tell the UI
which provider answered, and `GET /api/status` lists the providers and
whether each is usable, for the footer. An empty reply — which
reasoning models occasionally return — is retried once, then reported as an
error rather than shown as a blank answer.

**Follow-up actions and suggestions.** "Explain more simply" and "Go deeper"
send an ordinary follow-up (so the last turns resolve "that"), each
restating the grounding rule, since asking for "more" invited padding from
outside knowledge in testing. Suggested questions come from a separate
prompt over the overview sample; any failure just omits them.

**Sources.** The retrieved passages are returned with every answer,
independent of whether the model cites a page itself, so the user can
always check what the answer was (or wasn't) based on.

**Summary.** `summarize()` sends 10 evenly spaced chunks to a separate
summary prompt. A full map-reduce over every chunk would cost one LLM call
per chunk, which free-tier rate limits don't allow for longer PDFs.

## Technology Choices

| Choice | Why |
|--------|-----|
| **FastAPI + vanilla HTML/CSS/JS** | The smallest setup with full control over responsive layout; a SPA framework would add a build step and `node_modules` for one page. Same origin means no CORS. |
| **PyMuPDF** | Reliable page-level extraction, fast, no external binary. |
| **`all-MiniLM-L6-v2`** | Small (~80MB), fast on CPU, a well-known semantic-similarity baseline. |
| **FAISS `IndexFlatIP`, one per document** | Exact search is fine at a few documents × low thousands of chunks. Per-document indexes make removing one document a dict delete instead of re-embedding. |
| **OpenAI-compatible client via env vars** | No hardcoded provider; works with Groq (default), OpenAI or any compatible endpoint. |
| **No LangChain/LangGraph** | Every stage stays visible and explainable; see `DECISIONS.md` #1. |
| **Retrieval on concatenated queries, not LLM query rewriting** | Rewriting would double LLM calls per question on a rate-limited free tier. |

## Chunking Configuration

`pipeline.py` defaults — `chunk_size=800`, `chunk_overlap=150`, `top_k=4` —
come from the two-configuration evaluation in `DECISIONS.md` (small 300-char
chunks refused 4 of 5 answerable questions; 1000-char chunks answered all
5), and are backed by the measured retrieval evaluation (`retrieval_eval.py`:
Hit@4 0.59 at 300 characters vs 0.77 at 800 for the app's retriever). They are not exposed in the UI; `pipeline.ingest()`
and `pipeline.answer()` accept overrides, which `evaluate.py` uses to rerun
the comparison against any PDF.

## Evaluation Tooling

Two scripts sit beside the app and call the pipeline modules directly:

- `evaluate.py` — answer-level: runs a question set through the full
  pipeline (LLM included) under two chunk/top-k configurations and
  reports answers and sources side by side.
- `retrieval_eval.py` — retrieval-level, no LLM: Hit@1/Hit@4/MRR@10 on a
  labelled question set for a BM25 baseline, MiniLM, BGE-small and a
  BM25 + MiniLM hybrid (reciprocal rank fusion), across three chunk
  settings. Results: `reports/retrieval_eval.md`.

## Security Controls

| Threat | Control |
|---|---|
| Quota or CPU exhaustion by one client | Per-IP rate limits (`RATE_LIMITS`): LLM endpoints 10/min and 100/h, uploads 10/10 min; `X-Real-IP` is accepted only from a peer in the configured `TRUSTED_PROXIES` CIDRs |
| Memory exhaustion by a huge or aggregate document set | `MAX_CHUNKS_PER_DOC` (1500), `MAX_TOTAL_CHUNKS` (75,000 default), 5 docs/session, 50 sessions |
| Non-PDF uploads | `%PDF-` signature check in the first 1024 bytes |
| Hostile filenames | Basename only, control characters stripped, 120-character cap |
| XSS / clickjacking / injection of external resources | CSP `default-src 'self'` with no inline code, `frame-ancestors 'none'`, `X-Frame-Options: DENY`, `nosniff`; all dynamic text via `textContent` |
| Downgrade to HTTP | HSTS (1 year) over HTTPS; Nginx redirects plain HTTP, including requests to the raw IP |
| Reconnaissance | `/docs`, `/redoc`, `/openapi.json` disabled; Nginx `server_tokens off` |
| Leaking internals in errors | Generic message + reference code; details only in the server log |
| Cross-site request forgery | SameSite=Lax cookie, plus a middleware that refuses API POSTs whose `Origin` isn't this host or whose `Sec-Fetch-Site` is `cross-site` |
| Session cookie tampering | `__Host-session` over HTTPS: must be Secure, Path=/, no Domain, so no subdomain or HTTP response can set it |
| Oversized JSON bodies | 16KB cap (`Content-Length` check in middleware, and on the body actually read) |
| Rate-limit evasion by spoofing `X-Real-IP` | The header is honoured only when the immediate peer is inside `TRUSTED_PROXIES`; any other peer is rate-limited by its own address |
| Overload (smoothness) | Semaphores: 4 concurrent LLM calls, 2 concurrent ingestions; wait ≤ 20s, then 503 "busy" with `Retry-After` |
| Vulnerable dependencies | Dependabot updates; `pip-audit` in CI fails the build on any known advisory |
| Cross-session access (IDOR) | Document ids are looked up only inside the caller's own session (256-bit cookie) |
| Prompt injection via PDF text or filenames | Passages fenced and declared untrusted; fence tokens inside source text and names are neutralized before prompting |
| Weak retrieval / hallucination risk | The grounding prompt's exact-refusal rule; optional `RETRIEVAL_SCORE_FLOOR` (off by default) withholds weak passages; overview routing covers whole-document questions ("main focus of this resume" included) |
| Third-party data flow | Page loads nothing external; only the question + retrieved passages go to the LLM provider, disclosed on the upload screen |

## Failure Handling

| Situation | Handled in | User sees |
|---|---|---|
| Not a `.pdf`, or over 25MB | `main.py` (reads at most 25MB+1 bytes) | 400 / 413 with a message |
| PDF with no extractable text | `pdf_loader` → `ingest()` returns `None` | 422 "Couldn't extract any text…" |
| More than 5 documents | `main.py` | 400 |
| Invalid JSON / missing question | `main.py` `_json_body()` | 400 |
| Missing API key | `LLMConfigError` | 503, generic message |
| LLM network/HTTP failure, 429 past the retry cap | `LLMRequestError` | 502, "try again" |
| Anything unexpected | generic `except` in `main.py` | 500, generic message; details only in server log |
| Proxy returns an HTML error page | `app.js` `api()` | readable error, never a frozen screen |
| Stale cached frontend after a deploy | `Cache-Control: no-cache` + `?v=` asset URLs | always the current script |
| Question not answerable from the document | system prompt | the exact refusal string |
| Best passage below the (opt-in) similarity floor | `pipeline.gather_sources()` | no passages sent; the exact refusal string |
| Server-wide chunk budget exhausted | `main.py` `ingest()` (`MAX_TOTAL_CHUNKS`) | 503 "document memory is full" |
| Connection drops mid-stream | `llm_client._stream_deltas` → `LLMRequestError` | provider cooldown/failover before the first token; an `error` event after |
| Instructions embedded in a PDF | fenced passages + untrusted-content rule | reported as document text, not obeyed |

## Deployment

A single AWS EC2 `t3.small` behind Nginx (HTTPS via Let's Encrypt,
`client_max_body_size 25m`, 120s proxy timeouts), running the container with
`--restart unless-stopped` and the API key passed via `--env-file`. The
`Dockerfile` installs CPU-only PyTorch to avoid ~GBs of unused CUDA wheels.
Behind the Docker port mapping the app sees Nginx as the bridge gateway
rather than loopback, so `TRUSTED_PROXIES` must include it (see
`README.md`). Because sessions live in process memory, the app is designed for one
instance; scaling out would need a shared session store. Details and the
platform comparison are in `README.md` and `DECISIONS.md`.
