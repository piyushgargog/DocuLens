# AI Document Assistant

A Retrieval-Augmented Generation (RAG) tool that answers questions about
any PDF you upload — grounded strictly in that document's content, with
the exact source page shown for every answer.

## What it does

Upload a PDF, ask a question in plain language, and get back an answer
built only from the parts of the document that are actually relevant —
along with the page number(s) and passage text the answer came from. If
the document doesn't contain the answer, the assistant says so instead
of guessing.

## Problem it solves

Reading a long document to find one answer is slow, and keyword search
doesn't handle paraphrased questions. A plain LLM call without the
document, on the other hand, will confidently answer with information
that isn't actually in the source (hallucination). This project combines
retrieval (find the relevant passages) with generation (answer from only
those passages) so answers stay traceable back to the source text.

## Key features

- Upload any PDF and ask questions about it — no document-specific setup.
- **Multiple documents**: add up to 5 PDFs and ask across all of them;
  every source says which document and page it came from.
- **Follow-up questions**: the last few turns are sent with each question,
  so "how many moons does it have?" resolves "it" from the previous
  question. Answers still come only from retrieved passages.
- **Document summary**: one click per document for a short, page-cited
  summary.
- A page reload keeps your documents and conversation.
- A minimal, mobile-responsive chat interface — no frontend framework, no
  build step, just static HTML/CSS/JS served by the backend.
- Page-aware text extraction, so every retrieved passage keeps its
  source page number.
- Answers are grounded: the LLM is instructed to answer only from
  retrieved passages, and to say so explicitly when they don't contain
  the answer.
- Every answer carries its own sources panel showing the supporting page
  number(s), similarity scores, and passage text, so you can verify it
  against the original document.
- Retrieved document text is treated as untrusted data: it is fenced in
  the prompt and the model is instructed never to follow instructions
  embedded in a document (see [Security](#security-notes) below).
- Works with any OpenAI-compatible LLM API (OpenAI, Groq, or a local
  compatible endpoint) via environment variables — no vendor lock-in.

## Architecture

A small FastAPI backend wraps an explicit, individually-inspectable RAG
pipeline (no LangChain/LangGraph — see `DECISIONS.md` for why) and serves
a static frontend from the same origin, so no CORS setup is needed. The
pipeline itself — extraction, chunking, embedding, retrieval, prompting —
is unchanged from the project's original design; only the UI layer
changed (see `DECISIONS.md` for that migration and why).

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
[{text, page}, ...]  ──────────────► sentence-transformers ──► chunk embeddings
   │                                                                  │
   │                                                                  ▼
   │                                                         FAISS index (in memory,
   │                                                         held server-side per session)
   │
question (browser)
   │  POST /api/ask
   ▼
FastAPI ──► embed question ──► FAISS similarity search ──► top-k {text, page, score}
                                                                  │
                                                                  ▼
                                     prompt = system instruction + passages + question
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

Full design rationale is in `PROJECT_SPEC.md`, `ARCHITECTURE.md`, and
`IMPLEMENTATION_PLAN.md`. Every real engineering decision, failure, and
test result encountered while building this is logged chronologically in
`DECISIONS.md` — nothing there is fabricated.

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

## How the pipeline works

1. **Ingestion (once per uploaded document):** extract text per page,
   split each page's text into overlapping chunks (keeping track of
   which page each chunk came from), embed all chunks, and build a FAISS
   index over the embeddings. The resulting index is held in server
   memory, keyed by a session cookie set on the response.
2. **Query (once per question):** embed the question with the same
   model, retrieve the most similar chunks from the index, and build a
   prompt containing only those chunks (each labeled with its page
   number) plus an instruction to answer strictly from them — or say the
   document doesn't contain the answer.
3. **Response:** the LLM's answer is returned as JSON together with the
   retrieved passages and their page numbers, and the frontend renders
   both, so the answer can always be checked against the source.

## Installation

Requires Python 3.10+ (developed and tested on Python 3.14).

```bash
git clone https://github.com/piyushgargog/ai-document-assistant.git
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

```
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Any OpenAI-compatible chat completions endpoint works — change
`LLM_BASE_URL`/`LLM_MODEL` to use OpenAI or another compatible provider
instead of Groq. `.env` is gitignored; never commit real API keys.

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
2. Wait for indexing to finish — the document bar then shows the
   filename, page count, and chunk count.
3. Optionally click **+ Add PDF** to add more documents (up to 5).
   Questions then search all of them.
4. Ask a question in the chat box and press Enter (Shift+Enter for a
   newline).
5. Read the answer, then open the **Sources** disclosure under it to see
   exactly which document, page(s) and passage(s) it came from, with
   similarity scores.
6. Ask follow-up questions — the last 3 turns are sent along, so
   references like "it" or "that one" resolve.
7. Click **Summary** next to a document for a short summary of it.
8. Use **×** to remove one document, or **Clear all** to start over.

If the document contains no extractable text (empty, corrupt,
password-protected, or image-only without OCR), or the LLM API key is
missing/invalid, the app shows a clear error message instead of
crashing.

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
  explicit delimiters and the system prompt instructs the model to treat
  them as quoted data, never as instructions. Verified against a test
  PDF containing injected "ignore all previous instructions" and
  system-prompt-exfiltration payloads — the model reported the injected
  text as document content and refused the exfiltration attempt instead
  of obeying either (see `tests/test_prompt_injection.py`).
- **Secrets**: the API key is read only from the environment
  (`LLM_API_KEY`). It is never logged, rendered, or committed; `.env` is
  gitignored and only `.env.example` (placeholders) is tracked.
- **Session cookie**: the per-document session ID is an `httponly`,
  `samesite=lax` cookie set by the server — not readable from JavaScript,
  which limits exposure to XSS-based token theft.
- **Resource limits**: uploads are capped at 25MB and questions at 1000
  characters, since each upload is held in memory and embedded. Sessions
  expire from server memory after 2 hours of inactivity.
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

Tests that call a real LLM API skip automatically if `LLM_API_KEY` isn't
set — a GitHub Actions workflow runs the rest on every push/PR (see
`CONTRIBUTING.md`).

### Manual end-to-end / UI testing
The frontend itself has no automated coverage. It has been manually
verified in a real headless browser, at both desktop and mobile (375px)
viewport widths: upload → ingest → ask → answer with correct source page
→ expand sources → follow-up question → unanswerable question correctly
refused → remove document → re-upload, with zero browser console errors
and no horizontal overflow at mobile width.

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
questions, and gave one factually wrong numeric answer (BLEU 41.0
instead of the correct 41.8) — most likely because its 300-character
chunks split a dense results table mid-row. Config B answered all 5
correctly with accurate page citations. Both configs correctly refused
the genuinely unanswerable question. Full per-question results and
analysis are in `DECISIONS.md`. This is why the implementation's default
`chunk_size` (800) is close to config B, not the smaller value.

To run this comparison yourself against any PDF and question set:

```bash
python evaluate.py --pdf <path-to-pdf> --questions <path-to-questions.json> --output results.md
```

where `<path-to-questions.json>` is a JSON file containing a list of
question strings.

## Deployment

### Live deployment

Running at **https://ai-doc-assistant.duckdns.org** — an AWS EC2 `t3.small` (2 vCPU, 2GiB
RAM, Free Tier eligible), region `ap-south-1` (Mumbai), verified working
end-to-end (upload → indexing → grounded answer with source citation →
follow-up → correct refusal on an unanswerable question → remove/re-upload,
via a real browser against the public URL). Measured memory usage under
real load (a 15-page PDF, sentence-transformers model loaded, an actual
LLM call) peaked at 685MiB — comfortable headroom on the 2GiB instance.

Setup used: Docker (installed via the official Docker apt repository),
the repo's own `Dockerfile`, running as `docker run --restart
unless-stopped` (survives reboots automatically), with Nginx as a
reverse proxy in front (`client_max_body_size 25m` to match the app's
upload limit, generous proxy timeouts for LLM calls) forwarding to the
container's internal `127.0.0.1:8000`. HTTPS is terminated by Nginx
using a Let's Encrypt certificate (via Certbot) for the
`ai-doc-assistant.duckdns.org` domain (DuckDNS), with plain HTTP on
that domain redirected to HTTPS. Ports 80, 443 (public) and 22 (SSH,
restricted to a specific IP) are open in the security group — no
Elastic IP, load balancer, NAT gateway, or RDS were created; the
instance's own public IPv4 is used directly, with the domain pointed at
it via DuckDNS's dynamic DNS.

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
on the platform, and point it at port 8000. There's no platform-specific
configuration in this repo beyond the `Dockerfile` itself, deliberately —
picking one hosting provider's proprietary config format over a portable
container felt like the wrong default for a project meant to be run
anywhere. Note: the `Dockerfile` installs PyTorch's CPU-only wheel
explicitly (see `DECISIONS.md`) — without that, a plain `pip install`
of this project's dependencies on Linux pulls several hundred MB of
unused NVIDIA CUDA packages, which matters on a small instance's disk.

**Note on scale**: session state (the FAISS index per uploaded document)
lives in the process's memory — see Known limitations below. This is
fine for a single instance (which is exactly what's running above); it
is not designed to run behind a load balancer with multiple replicas
without a shared session store.

## Known limitations

- Up to 5 documents per session; each is searched separately and the
  results merged, which is fine for a handful of PDFs, not a large corpus.
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
- Uploads are capped at 25MB, and scanned/image-only PDFs with no
  embedded text layer will not extract any text (no OCR step).
- Chunking is character-based, not sentence/semantic-boundary aware —
  simple and predictable, but can occasionally split content (e.g. a
  table row) across a chunk boundary, as observed during evaluation
  above.
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

- **Query rewriting** — have the LLM rewrite a follow-up into a
  standalone question before retrieval, instead of concatenating it with
  the previous one.
- **Full-document summary** — map-reduce over every chunk once a
  provider with a higher rate limit is used.
- **Smarter chunking** — sentence- or section-boundary-aware chunking
  instead of a fixed character window, to reduce the table-splitting
  failure mode observed in evaluation.
- **Reranking** — a lightweight cross-encoder reranking step over the
  initial retrieval results, for higher-precision passage selection on
  larger documents.
- **Shared session store** — swap the in-memory session dict for Redis
  (or similar) if this ever needs to run behind multiple replicas.
