# DocuLens — An AI Powered Document Assistant

A Retrieval-Augmented Generation (RAG) tool that answers questions about
the documents you upload — grounded strictly in their content, with the exact
source page and passage shown beside every answer.

**Live:** https://doculens.duckdns.org — **Latest release:** v3.6.2

## What it does

Upload one or more documents (PDF, Word, text or Markdown), ask a question
in plain language, and get back an answer built only from the passages that are actually relevant — with
the document, page number and passage text each answer came from. If the
documents don't contain the answer, the assistant says so instead of
guessing.

## Problem it solves

Reading a long document to find one answer is slow, and keyword search
doesn't handle paraphrased questions. A plain LLM call without the
document, on the other hand, will confidently answer with information
that isn't actually in the source (hallucination). This project combines
retrieval (find the relevant passages) with generation (answer from only
those passages) so answers stay traceable back to the source text.

## Key features

**Ask and verify**
- Upload any document and ask questions about it, with no document-specific
  setup. **PDF, Word (.docx), text and Markdown** are accepted; non-PDF files
  are split into even pages so citations stay meaningful, and **scanned PDFs are
  OCR'd** (Tesseract).
- **Multiple documents**: add up to 5 and ask across all of them, or tick the
  ones to search. Every source says which document and page it came from.
- **Sources beside every answer**: on a wide screen each answer shows its source
  passages as cards; page references become clickable `p. 3` chips that light
  up the exact passage. On a phone the cards fold under the answer.
- **Follow-up questions**: the last few turns are sent with each question, so
  "how many moons does it have?" resolves "it". Answers still come only from
  retrieved passages.
- **Document summary** (one click, page-cited) and **tailored "Try asking"
  starters** written from the document.
- **Answers stream in** as they are written, with a **Stop** button (or `Esc`);
  a stopped answer is kept on screen but not used as context. A page reload
  keeps your documents and conversation. **Copy** any answer with its sources,
  **export** the conversation as Markdown, drop a file anywhere, `/` jumps to
  the question box.
- **Rich, readable answers**: proper Markdown (headings, lists, tables, code)
  with inline citation chips, built safely with no `innerHTML`.
- **Shows the model's thinking**: when a model exposes its reasoning it streams
  into a collapsible "Thinking…" panel that folds to "Thought for Ns". It is
  display-only: never saved to the history or fed back to the model.

**Grounded, not guessed**
- The LLM is told to answer only from the retrieved passages and to say so
  explicitly when they don't contain the answer (one exact refusal sentence).
- **Hybrid retrieval**: BM25 keyword scoring fused with embedding similarity
  (reciprocal rank fusion), the retriever that measured best in
  [`retrieval_eval.py`](#retrieval-evaluation-measured): Hit@4 0.77 → **0.82**,
  MRR 0.56 → **0.70**. Whole-document questions ("summarize this", "what is the
  main focus of this resume?") use an ordered sample instead of similarity
  search.
- Retrieved document text is **treated as untrusted data**: fenced in the prompt
  (fence tokens in uploaded text and filenames are neutralised), earlier chat
  turns are labelled untrusted, and the model is told never to follow
  instructions embedded in a document (see [Security](#security-notes)).
- An optional similarity floor (`RETRIEVAL_SCORE_FLOOR`, **off by default**) can
  withhold weak passages; see [Score-floor calibration](#score-floor-calibration-v360-revised-in-v361)
  for why it is opt-in.

**Reliable and safe to run**
- **Several AI providers with automatic failover**: Groq and Hugging Face
  (`gpt-oss-120b`), OpenRouter (Nemotron 3 Super 120B, free tier) and NVIDIA
  (`gpt-oss-20b`); Google AI Studio (Gemini) is opt-in. If one is rate limited,
  out of quota, down or misconfigured (including a connection that drops
  mid-stream), the question goes to the next and the failing one cools down.
  Each answer shows which provider and model wrote it; the footer shows which
  are available now. Any OpenAI-compatible endpoint can be plugged in.
- **Load control and limits**: at most 4 LLM calls and 2 ingestions at once
  (extra requests wait briefly, then get a clear "busy" reply), per-IP rate
  limits that are correct behind a reverse proxy (`TRUSTED_PROXIES`), and a
  server-wide index budget (`MAX_TOTAL_CHUNKS`).
- **Security**: strict CSP and security headers, cross-site request blocking
  (Origin / `Sec-Fetch-Site`), a `__Host-` session cookie over HTTPS with a
  sliding 2-hour expiry, a 16KB JSON cap, `pip-audit` and CodeQL in CI. Uploads
  are never written to disk.

**Interface**
- A quiet, Claude-style **"Calm Light" design**: ink on warm near-white, one
  terracotta accent, Newsreader serif over Inter, light and dark themes,
  keyboard-accessible and mobile-responsive. Motion is restrained and responds
  to what you do (almost all of it switches off under reduced-motion settings;
  the quick page settle-in and the footer heartbeat deliberately stay). A
  full footer carries the author credit, repository and release links, a
  privacy dialog and the running version. No frontend framework and no build
  step: static HTML/CSS/JS served by the backend. Rationale in
  [`DESIGN.md`](DESIGN.md).

**Release history** is on the [GitHub Releases](https://github.com/piyushgargog/DocuLens/releases)
page, and every engineering decision, failure and fix (including mistakes, like
the v3.6.0 similarity floor) is logged chronologically in
[`DECISIONS.md`](DECISIONS.md).

## Architecture

A small FastAPI backend wraps an explicit, individually-inspectable RAG
pipeline (no LangChain/LangGraph — see `DECISIONS.md` for why) and serves
a static frontend from the same origin, so no CORS setup is needed. The
core stages — extraction, chunking, embedding, retrieval, prompting — are
the original design; multi-document retrieval, follow-up context and
summaries were added on top without changing the single-document path
(tests enforce this, so the evaluation below still holds). Full detail is
in [`ARCHITECTURE.md`](ARCHITECTURE.md).

```
PDF upload (browser)
   │  POST /api/ingest
   ▼
FastAPI (main.py)
   │  PyMuPDF (page-aware extraction)
   ▼
[(page_num, page_text), ...]
   │  chunker.py (fixed size / overlap)
   ▼
[{text, page, doc}, ...]  ─────────► sentence-transformers ──► chunk embeddings
   │                                                                  │
   │                                                                  ▼
   │                                                  one FAISS index per document
   │                                                  (in memory, up to 5 per session)
   │
question (browser)
   │  POST /api/ask
   ▼
FastAPI ──► embed question (+ previous question for follow-ups)
        ──► search every document, merge ──► top-k {text, page, doc, score}
                                                                  │
                                                                  ▼
                  prompt = system rules + fenced passages + last 3 turns + question
                                                                  │
                                                                  ▼
                                         LLM API call (any OpenAI-compatible endpoint)
                                                                  │
                                                                  ▼
                                          {answer, sources} JSON ──► rendered by app.js
```

| Component | File | Tech |
|---|---|---|
| PDF loader | `pdf_loader.py` | PyMuPDF |
| Chunker | `chunker.py` | plain Python, character sliding window |
| Embedder | `embedder.py` | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | `vector_store.py` | FAISS `IndexFlatIP` |
| LLM client | `llm_client.py` | `requests`, OpenAI-compatible REST |
| Orchestration | `pipeline.py` | ties the above together |
| Backend / API | `main.py` | FastAPI, in-memory per-session state |
| Frontend | `static/index.html`, `static/style.css`, `static/app.js` | vanilla HTML/CSS/JS, no framework |
| Evaluation | `evaluate.py` | reproducible two-config comparison script |
| Retrieval evaluation | `retrieval_eval.py` | Hit@k / MRR vs a BM25 baseline and two embedding models |
| Score-floor calibration | `score_floor_eval.py` | how often each similarity floor would refuse answerable vs off-topic questions |
| Design | `DESIGN.md` | the frontend's written design direction |

Full design rationale is in `PROJECT_SPEC.md`, `ARCHITECTURE.md`, and
`IMPLEMENTATION_PLAN.md`. Every real engineering decision, failure, and
test result encountered while building this is logged chronologically in
`DECISIONS.md` — nothing there is fabricated.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /api/ingest` | Upload a document (PDF, .docx, .txt or .md; multipart) into the session |
| `POST /api/ask/stream` | Ask; answer as server-sent events: `sources`, optional `reasoning`, `route`, `token`…, `done` / `error` |
| `POST /api/ask` | Same, as one JSON response |
| `POST /api/summary` | Summary of one document |
| `POST /api/suggestions` | Starter questions for one document |
| `POST /api/remove` | Remove one document, or everything |
| `GET /api/session` | Documents and conversation, for restoring after a reload |

`/api/ask*` accept an optional `doc_ids` list to search only some documents.

## Technology stack

- **Python**
- **FastAPI** + **uvicorn** — a minimal HTTP API and static file server,
  no heavier web framework needed for this project's scope
- **Vanilla HTML/CSS/JS** frontend — no React/Vue/build tooling; a single
  page is all this needs
- **PyMuPDF** — page-aware PDF text extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`) — text embeddings
- **FAISS** — vector similarity search
- Any **OpenAI-compatible LLM API** (OpenAI, Groq, etc.) via a minimal
  `requests`-based client — no heavyweight SDK or agent framework
- **Docker** (optional) for deployment — see [Deployment](#deployment)

## What the LLM does (and why this isn't a text extractor)

Retrieval finds *where* the document talks about something. The LLM turns
those passages into an answer to *your* question:

- **Explains in its own words, at your level.** "Explain multi-head
  attention" and "Explain it more simply" give different answers from the
  same passages; the second drops the formulas and uses everyday language.
- **Combines passages.** "How is self-attention better than recurrent
  layers?" pulls a complexity table and two paragraphs into one three-point
  comparison — no single passage says that.
- **Answers whole-document questions.** "What is this paper about?" matches
  no particular passage, so such questions are recognised and routed to an
  overview of the document (its opening plus evenly spaced passages) for the
  LLM to condense — before this, similarity search returned the reference
  list and the assistant refused.
- **Says what's missing.** If the documents answer only part of a question,
  it answers that part and names what isn't covered; if they answer none of
  it, it refuses rather than guessing.
- **Suggests where to start.** After upload it reads a sample of the
  document and proposes questions worth asking.

What it may not do is add knowledge of its own: every claim must come from
the passages and carry a page citation, and each answer shows those
passages beside it so you can check.

## How the pipeline works

1. **Ingestion (once per uploaded PDF):** extract text per page, split
   each page into overlapping chunks tagged with their page and document,
   embed them, and build that document's FAISS index. It is held in server
   memory in the browser's session (identified by a cookie), alongside up
   to four other documents.
2. **Query (once per question):** embed the question with the same model,
   search every loaded document and keep the overall best passages. For a
   follow-up, the previous question is searched too, so "it" resolves.
   The prompt contains only those passages (labeled with document and
   page), the last few turns for context (labelled as untrusted
   references), and an instruction to answer strictly from the passages —
   or say the documents don't contain the answer. If even the best passage
   is clearly unrelated (below the calibrated similarity floor), no
   passages are sent at all and the model must refuse.
3. **Response:** the answer is returned with the retrieved passages; the
   frontend shows them as margin notes and turns page references in the
   answer into clickable tabs, so every claim can be checked against the
   source.

## Installation

Requires Python 3.10+ (developed and tested on Python 3.14).

```bash
git clone https://github.com/piyushgargog/DocuLens.git
cd ai-document-assistant
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Environment configuration

Copy `.env.example` to `.env` and fill in your LLM API key:

```bash
# Windows (Command Prompt)
copy .env.example .env
# Windows (PowerShell) / macOS / Linux
cp .env.example .env
```

Set at least one provider key. Every key you add is another provider
the app can fall back to:

```
GROQ_API_KEY=...          # https://console.groq.com/keys
OPENROUTER_API_KEY=...    # https://openrouter.ai/keys
NVIDIA_API_KEY=...        # https://build.nvidia.com
HF_TOKEN=...              # https://huggingface.co/settings/tokens
```

`LLM_PROVIDERS` sets the order (default `groq,openrouter,nvidia,huggingface`; add `google` and
`GEMINI_API_KEY` to use Gemini),
`<NAME>_MODEL` / `<NAME>_BASE_URL` override a provider's defaults, and
`LLM_FALLBACK_MODEL` (e.g. `openai/gpt-oss-20b`) adds a second Groq model
at the end of the chain, since each Groq model has its own daily quota.
Older `.env` files with `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` keep
working: they configure the first slot, and can point it at any
OpenAI-compatible endpoint. `.env` is gitignored; never commit real API
keys. See `.env.example` for everything.

Optional production settings (all have safe defaults):

| Variable | Default | Purpose |
|---|---|---|
| `TRUSTED_PROXIES` | `127.0.0.0/8,::1/128` | CIDRs whose `X-Real-IP` header is trusted for rate limiting. Behind a Docker port-publish set the bridge gateway, e.g. `172.17.0.1/32,127.0.0.0/8,::1/128`. Never `0.0.0.0/0`. |
| `MAX_TOTAL_CHUNKS` | `75000` | Server-wide cap on indexed chunks (≈3 KB each); uploads that don't fit get a 503. Lower it on small hosts. |
| `RETRIEVAL_SCORE_FLOOR` | `0` (off) | Opt-in: best-hit cosine similarity below which a non-overview question is refused without calling the model. Off by default because it wrongly refuses questions on short documents; measure it with `score_floor_eval.py` on your own documents before enabling it. |
| `PREFETCH_MODEL` | `0` | `1` loads the embedding model at startup instead of on the first upload. |
| `LOG_LEVEL`, `SENTRY_DSN` | `INFO`, empty | Logging verbosity; opt-in error reporting. |

## How to run

```bash
uvicorn main:app --reload
```

Then open `http://localhost:8000` in your browser. `--reload` is for
local development (auto-restarts on code changes); drop it for anything
resembling production use.

## Usage

1. Upload a PDF (click the upload area, or drag a file onto it). 25MB
   limit.
2. Wait for the PDF to be read — it then appears under **Your documents**
   with its page count.
3. Optionally click **Add another PDF** to add more documents (up to 5).
   Questions then search all of them.
4. Ask a question in the box at the bottom and press Enter (Shift+Enter
   for a newline).
5. Read the answer. Its retrieved passages sit in the margin beside it
   (or behind **Show sources** on a phone), each with its document, page
   and similarity score. Click a yellow page tab (e.g. **p. 3**) to
   highlight the passage it came from; click a margin note to expand it.
6. Ask follow-up questions — the last 3 turns are sent along, so
   references like "it" or "that one" resolve. Under each answer,
   **Explain more simply** and **Go deeper** ask the assistant to rework it.
   After the first upload, **Try asking** offers suggested questions.
7. Click **Summarize** under a document for a short summary of it.
8. Use **×** to remove one document, or **Remove all** to start over.

Your PDF is never saved to disk. Its extracted text is kept in server
memory only while you use it, and is deleted when you remove the document,
after 2 hours of inactivity, or when the server restarts.

When an answer arrives, the conversation scrolls to your question so the
answer reads from its start.

Scanned / image-only PDFs (no text layer) are run through OCR
(Tesseract) automatically, so a scan can be read too. If a document still
yields no text (empty, corrupt, password-protected, or a scan OCR can't make
out), or the LLM API key is missing/invalid, the app shows a clear error
message instead of crashing.

### How source citations work

Every retrieved passage is tagged with the page number it came from
during chunking (`chunker.py`). When the LLM answers, the app shows the
top-k retrieved passages alongside that specific answer — independent of
whether the LLM explicitly cites a page in its own text — so you can
always see which parts of the document the answer is (or isn't) actually
grounded in, along with a similarity score for each. The frontend inserts
all document/answer text with `textContent`, never `innerHTML`, so
nothing from the document or the model can inject markup into the page.

## Security notes

This is a document QA tool that feeds untrusted file content into an
LLM, so a few things are handled deliberately:

- **Indirect prompt injection**: retrieved passages are fenced inside
  explicit delimiters and delimiter tokens inside uploaded text and filenames
  are neutralized before prompting. The system prompt treats passages as
  quoted data, never as instructions. Verified against injected "ignore all
  previous instructions" and system-prompt-exfiltration payloads (see
  `tests/test_prompt_injection.py`).
- **Secrets**: provider API keys are read only from the environment
  (`GEMINI_API_KEY`, `GROQ_API_KEY`/`LLM_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`,
  `HF_TOKEN`). `/api/status` reports provider names, models and
  availability only — never keys or URLs. It is never logged, rendered, or committed; `.env` is
  gitignored and only `.env.example` (placeholders) is tracked.
- **Session cookie**: the session ID is an `httponly`, `samesite=lax`
  cookie, marked `Secure` whenever the site is served over HTTPS — not
  readable from JavaScript, which limits exposure to XSS-based token theft.
  Active API requests refresh its 2-hour browser expiry without changing the
  session ID. Sessions are anonymous bearer sessions, not user accounts.
- **Errors**: clients get a fixed message and a proper status code;
  exception details are logged on the server only (CodeQL
  `py/stack-trace-exposure`).
- **Uploaded PDFs are not stored**: nothing is written to disk by the
  app. Starlette's temporary spool file for a large upload is closed as soon
  as the bytes are read; only the extracted text and embeddings are kept, in
  memory, and they are deleted when the user removes the document or after
  2 hours of inactivity. A background task enforces that deadline every 5
  minutes even when no one is using the site, and a server restart clears
  everything. The upload screen tells users this.
- **What leaves the server**: to write an answer, the question and the
  retrieved passages are sent to one AI provider (whichever of Groq,
  OpenRouter, NVIDIA or Hugging Face is configured and available); the
  upload screen and the Privacy dialog say so, and each answer names the
  provider that wrote it. Fonts are self-hosted, so the page
  itself makes no requests to other sites.
- **Response headers**: a strict Content-Security-Policy (everything from
  `'self'`, no inline scripts or styles, no framing), `nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, a restrictive
  `Permissions-Policy`, and HSTS over HTTPS. The FastAPI API docs
  (`/docs`, `/redoc`, `/openapi.json`) are disabled.
- **Rate limits** per client IP: 10 LLM-backed requests a minute and 100 an
  hour (questions, summaries, suggestions), 10 uploads per 10 minutes —
  so one script can't exhaust the shared LLM quota or the CPU. `X-Real-IP` is
  used only when the immediate peer belongs to a CIDR in `TRUSTED_PROXIES`
  (loopback-only by default; configure the Docker bridge explicitly).
- **Uploads**: the file must start like a PDF (`%PDF-`), not just be named
  `.pdf`; filenames are cut to 120 characters and stripped of control
  characters; a document over ~360 pages of text is rejected *before*
  embedding, so one upload can't exhaust memory.
- **Errors**: unexpected failures return a generic message with a short
  reference code that matches the server log, never internals.
- **Resource limits**: uploads are capped at 25MB (never read past the
  limit), questions at 1000 characters, 5 documents per session, 50
  sessions per server, and sessions expire after 2 hours of inactivity. A
  best-effort `MAX_TOTAL_CHUNKS` budget (75,000 by default) also limits the
  aggregate in-memory index footprint; tune it down for small hosts.
- **Grounded refusal**: the model must answer only from the retrieved passages
  or return one exact refusal string. An optional similarity floor
  (`RETRIEVAL_SCORE_FLOOR`, off by default) can additionally withhold weak
  passages; it is opt-in because similarity alone can't tell a short or
  pronoun-heavy question from an unrelated one. Whole-document overview
  questions use a document-order sample instead of similarity search.
- **Rendering**: all dynamic content is inserted via `textContent` (see
  above), never raw HTML or markdown interpretation.

## Evaluation methodology

Every test result described here was actually run; none is assumed or
fabricated — see `DECISIONS.md` for the full chronological log.

### Automated testing
A committed `pytest` suite (`tests/`) covers PDF extraction, chunking,
embeddings, FAISS retrieval, the LLM client, the full pipeline,
prompt-injection resistance, and the FastAPI HTTP layer (upload, session
handling, error responses). Run it yourself:

```bash
pip install -r requirements-dev.txt
pytest -v
```

The suite has 153 tests (a handful that call a real LLM skip without a
key); the new v3.6.0 code is covered by tests for proxy-CIDR trust, the chunk
budget, fence-token neutralisation, history labelling, the (opt-in) abstention gate,
mid-stream network errors, cookie refresh and startup prefetch.

Tests that call a real LLM API skip automatically if `LLM_API_KEY` isn't
set — a GitHub Actions workflow runs the rest on every push/PR (see
`CONTRIBUTING.md`).

### Browser testing
The frontend has no unit tests; every release is instead driven end to
end in a real headless browser against both a local server and the live
site, at desktop and 375px mobile widths and in light and dark themes:
upload → answer with a page tab → clicking the tab highlights the right
margin note → follow-up question → unanswerable question refused →
second PDF → cross-document answer → summary → page reload restores the
session → remove one / remove all. Each release is shipped only with zero
console errors, zero failed requests and no horizontal overflow on
mobile. Keyboard access (Tab to the upload area, Enter to open the file
picker) is checked too.

### Real-world validation
A minimal synthetic PDF is bundled (`sample_docs/sample.pdf`) purely as
a quick demo/smoke-test document. To validate the pipeline on realistic,
non-synthetic document structure, a public real-world PDF — the paper
"Attention Is All You Need" (Vaswani et al., 2017), 15 pages of real
body text, section headings, data tables, and references — was used for
a more thorough evaluation:

- **Question set:** 6 questions independently verified against the
  extracted text before running the test — 5 answerable, 1 genuinely
  unanswerable from the document (asks for a training cost in US
  dollars; the paper only ever reports cost in FLOPs).
- **Two configurations compared** via `evaluate.py`:
  - Config A: chunk_size=300, chunk_overlap=50, top_k=3
  - Config B: chunk_size=1000, chunk_overlap=200, top_k=5

**Result:** Config A incorrectly refused to answer 4 of the 5 answerable
questions. It answered the English-French BLEU question with 41.0 where
Config B said 41.8 — both are in the paper (41.8 in the abstract and
Table 2, 41.0 in the Section 6.1 text on page 8), which was first logged
as a mis-read and later corrected. Config B answered all 5 with accurate
page citations. Both configs correctly refused
the genuinely unanswerable question. Full per-question results and
analysis are in `DECISIONS.md`. This is why the implementation's default
`chunk_size` (800) is close to config B, not the smaller value.

To run this comparison yourself against any PDF and question set:

```bash
python evaluate.py --pdf <path-to-pdf> --questions <path-to-questions.json> --output results.md
```

where `<path-to-questions.json>` is a JSON file containing a list of
question strings.

### Retrieval evaluation (measured)

The comparison above judges whole answers. `retrieval_eval.py` measures
the step that decides what the LLM gets to see — retrieval — with
standard information-retrieval metrics and no LLM calls, so it is free,
deterministic and fully reproducible:

- **Labelled set:** 39 questions on the same 15-page paper
  (`sample_docs/retrieval_eval_set.json`), each with its gold page and a
  short evidence phrase copied from that page. A retrieved chunk counts as
  relevant only if it contains the phrase; the script refuses to run if
  any phrase is missing from its page, so labels can't silently drift.
- **Retrievers:** a **BM25** keyword baseline (implemented in the script),
  **MiniLM** (`all-MiniLM-L6-v2`, the app's model), **BGE-small**
  (`BAAI/bge-small-en-v1.5`, a second Hugging Face model) and a **Hybrid**
  of BM25 + MiniLM fused with reciprocal rank fusion.
- **Metrics:** Hit@1 and Hit@4 (4 = the app's top_k, i.e. what the LLM
  receives), MRR@10, and Page-Hit@4 — across three chunking settings.

![Hit@4 by retriever and chunking configuration](reports/retrieval_eval.svg)

At the app's default chunking (800/150):

| Retriever | Hit@1 | Hit@4 | MRR@10 |
|---|---|---|---|
| BM25 (baseline) | 0.64 (25/39) | **0.87** (34/39) | **0.75** |
| MiniLM (used by the app) | 0.38 (15/39) | 0.77 (30/39) | 0.56 |
| BGE-small | 0.59 (23/39) | 0.77 (30/39) | 0.68 |
| Hybrid (BM25 + MiniLM) | 0.56 (22/39) | 0.82 (32/39) | 0.70 |

**What it shows:**
- **Chunk size matters for every retriever:** 300-character chunks are
  the worst setting for all four (MiniLM Hit@4 0.59 vs 0.77 at 800) — the
  same direction as the answer-level comparison above.
- **The keyword baseline beat the app's embedding model on this
  document**, especially at rank 1 (Hit@1 0.64 vs 0.38). BGE-small ranks
  the right chunk first far more often than MiniLM (0.59 vs 0.38) at the
  cost of ~2× the embedding time. Fusing BM25 with MiniLM lifts MiniLM's
  Hit@4 from 0.77 to 0.82 and MRR from 0.56 to 0.70.
- **Caveats, stated plainly:** the questions were written by reading the
  document, so they share its wording — which favours BM25 and likely
  overstates its lead over embeddings for real users who paraphrase. With
  39 questions one question is 0.026, so gaps under ~0.05 are within
  noise. It is one document.

**Decision:** the app is unchanged for now. Hybrid retrieval is the
evidence-backed next step — it keeps the embedding model's handling of
paraphrase while recovering exact-term matches — but it should be
re-measured on the organizers' document first, since this set's wording
flatters keyword search. Full results, per-question misses and timings:
[`reports/retrieval_eval.md`](reports/retrieval_eval.md).

```bash
python retrieval_eval.py --output reports/retrieval_eval.md --chart reports/retrieval_eval.svg
```

### Score-floor calibration (v3.6.0, revised in v3.6.1)

`RETRIEVAL_SCORE_FLOOR` withholds passages when the best hit is too dissimilar
to the question. `score_floor_eval.py` scores the best top-4 hit for answerable
questions, off-topic questions and cross-document questions and shows how many
each floor would refuse (no LLM calls).

**What happened, stated plainly.** On the two sample documents the floor looked
safe: off-topic and cross-document questions top out at 0.28 (median about 0.1)
while answerable ones have a median of 0.5-0.65, and 0.25 wrongly refused ~3% of
answerable questions. v3.6.0 shipped it on at 0.25. The first real upload broke
it: on a one-page resume, answerable questions scored **0.06-0.27** ("where did he
work?" 0.06, "education?" 0.19, "main focus?" 0.21, "skills?" 0.25), the same
range as an unrelated question (0.05). Short documents and generic or
pronoun-heavy questions simply have low cosine similarity to their own answers,
so no threshold separates them from off-topic ones. The two sample documents were
too few and too uniform to reveal that.

**Decision (v3.6.1):** the floor is **off by default** (`RETRIEVAL_SCORE_FLOOR=0`).
The control that matters is the grounding prompt's exact-refusal rule, which
handled the resume correctly: "capital of France?" was refused, and the "main
focus" question was answered once it was routed as a whole-document question.
Enabling the floor remains possible for corpora where it measures well; run
`score_floor_eval.py` on your own documents and check that answerable questions
are almost never refused before turning it on.

```bash
python score_floor_eval.py --output reports/score_floor_eval.md
```

## Deployment

### Live deployment

Running at **https://doculens.duckdns.org** — an AWS EC2 `t3.small` (2 vCPU, 2GiB
RAM, Free Tier eligible), region `ap-south-1` (Mumbai), verified working
end-to-end (upload → indexing → grounded answer with source citation →
follow-up → correct refusal on an unanswerable question → remove/re-upload,
via a real browser against the public URL). Measured memory usage under
real load (a 15-page PDF, sentence-transformers model loaded, an actual
LLM call) peaked at 685MiB — comfortable headroom on the 2GiB instance.

Setup used: Docker (installed via the official Docker apt repository),
the repo's own `Dockerfile` for the base image (updates since v1.2.0 add
a thin code layer on top of it, because the instance's small disk can't
hold a full parallel rebuild — see `DECISIONS.md`), running as `docker run --restart
unless-stopped` (survives reboots automatically), with Nginx as a
reverse proxy in front (`client_max_body_size 25m` to match the app's
upload limit, generous proxy timeouts for LLM calls) forwarding to the
container's internal `127.0.0.1:8000`. HTTPS is terminated by Nginx
using a Let's Encrypt certificate (via Certbot) for the
`doculens.duckdns.org` domain (DuckDNS), with plain HTTP on
that domain redirected to HTTPS. Ports 80, 443 (public) and 22 (SSH,
restricted to a specific IP) are open in the security group — no
Elastic IP, load balancer, NAT gateway, or RDS were created; the
instance's own public IPv4 is used directly, with the domain pointed at
it via DuckDNS's dynamic DNS.

The container is started with `--log-opt max-size=10m --log-opt max-file=3` so
Docker's default unlimited log files can't fill the small root disk.

`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` were set via a `.env` file
transferred directly to the instance over `scp` and passed to the
container with `--env-file` — never committed, never part of the Docker
image, never printed to any log.

### Deploying it yourself, anywhere

The app is a standard FastAPI ASGI app, deployable anywhere that can run
one. A `Dockerfile` is included:

```bash
docker build -t ai-document-assistant .
docker run -p 8000:8000 --env-file .env ai-document-assistant
```

Any host that runs containers (a plain VM, Render, Railway, Google Cloud
Run, etc.) works the same way: build the image, set `LLM_API_KEY` (and
optionally `LLM_BASE_URL`/`LLM_MODEL`) as environment variables/secrets
on the platform, and point it at port 8000. For a reverse proxy, set
`TRUSTED_PROXIES` to the proxy network CIDR (for example
`172.17.0.0/16,127.0.0.0/8,::1/128`); never use `0.0.0.0/0`. Set
`PREFETCH_MODEL=1` when predictable first-request latency is more important
than startup time. `MAX_TOTAL_CHUNKS` can be lowered on memory-constrained
hosts. There is no platform-specific configuration in this repo beyond the
`Dockerfile` itself, deliberately — picking one hosting provider's
proprietary config format over a portable container felt like the wrong
default for a project meant to be run anywhere. Note: the `Dockerfile`
installs PyTorch's CPU-only wheel explicitly (see `DECISIONS.md`) — without
that, a plain `pip install` of this project's dependencies on Linux pulls
several hundred MB of unused NVIDIA CUDA packages, which matters on a small
instance's disk.

**Note on scale**: session state (the FAISS index per uploaded document)
lives in the process's memory — see Known limitations below. This is
fine for a single instance (which is exactly what's running above); it
is not designed to run behind a load balancer with multiple replicas
without a shared session store.

## Known limitations

- Up to 5 documents per session; each is searched separately and the
  results merged, which is fine for a handful of PDFs, not a large corpus.
  Across all sessions the server indexes at most `MAX_TOTAL_CHUNKS` chunks;
  that budget is checked before and after embedding but is a best-effort soft
  limit (two uploads racing can briefly overshoot), not a hard quota.
- Even a refused answer lists the passages that were retrieved for it (the best
  four by similarity, whatever they scored), so "I could not find the answer"
  can still show source cards: they are what was searched, not evidence. The
  optional similarity floor that would hide them is off by default because it
  also hid real answers on short documents.
- Only the last 3 conversation turns are used for follow-ups, and the
  follow-up retrieval simply combines the previous and current question
  (no LLM query rewriting, to avoid an extra API call per question).
- Summaries are built from 10 evenly spaced excerpts, not the whole
  document, so details between them can be missed.
- Session state lives in server process memory, keyed by a cookie: it is
  lost on server restart, isn't shared across multiple instances/replicas
  of the app, and is pruned after 2 hours of inactivity. This is a
  deliberate simplicity tradeoff for a project at this scale, not an
  oversight — a production multi-instance deployment would need a shared
  store (e.g. Redis) instead.
- Uploads are capped at 25MB. Scanned / image-only PDFs are OCR'd
  (Tesseract, English, up to 30 pages); handwriting or low-quality scans may
  still not be readable.
- Chunking is character-based, not sentence/semantic-boundary aware —
  simple and predictable, but a sentence or table row can be cut at a
  chunk boundary, and small chunks measurably hurt retrieval (Hit@4 0.59
  at 300 characters vs 0.77 at 800 for the app's retriever — see
  [Retrieval evaluation](#retrieval-evaluation-measured)).
- Retrieval quality depends on the embedding model
  (`all-MiniLM-L6-v2`) — a small, fast, general-purpose model chosen for
  reliability at this project's scale, not maximum retrieval accuracy.
- LLM API rate limits (e.g. Groq's free-tier tokens-per-minute cap) can
  slow down rapid, repeated evaluation runs; `llm_client.py` retries
  automatically with backoff, but very fast bulk evaluation may still be
  gated by provider limits.
- The live deployment is a single EC2 instance with no monitoring,
  auto-restart-on-crash beyond Docker's own `--restart unless-stopped`,
  or backup — appropriate for a portfolio demo, not for anything
  requiring uptime guarantees.

## Possible future improvements

- **Streaming ingestion progress** — report pages read and chunks
  embedded while a large PDF is indexed.
- **Query rewriting** — have the LLM rewrite a follow-up into a
  standalone question before retrieval, instead of concatenating it with
  the previous one.
- **Full-document summary** — map-reduce over every chunk once a
  provider with a higher rate limit is used.
- **Smarter chunking** — sentence- or section-boundary-aware chunking
  instead of a fixed character window, so facts aren't cut at a chunk
  boundary.
- **Reranking** — a lightweight cross-encoder reranking step over the
  hybrid results, for higher-precision passage selection on larger
  documents (would need re-measuring against the 2GB memory budget).
- **Shared session store** — swap the in-memory session dict for Redis
  (or similar) if this ever needs to run behind multiple replicas.
