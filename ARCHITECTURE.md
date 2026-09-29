# Architecture — AI Document Assistant

## Overview

A small FastAPI backend implementing a classic Retrieval-Augmented
Generation (RAG) pipeline, built from explicit, individually-inspectable
steps rather than a framework's black-box chain. This is a deliberate
design choice (see `DECISIONS.md`): each pipeline stage is a plain
function you can point to and explain, rather than logic hidden behind a
framework's abstraction. The UI layer was originally built with
Streamlit and later replaced with a minimal FastAPI + static-frontend
setup for a better, mobile-responsive user experience (see `DECISIONS.md`
for that migration) — the pipeline stages below (2–7) were untouched by
that change.

## Components

1. **Backend / API** (`main.py`, FastAPI)
   Exposes `POST /api/ingest`, `POST /api/ask`, `POST /api/summary`,
   `POST /api/remove`, `GET /api/session`, and serves the static frontend
   from the same origin (no CORS needed). Holds a `Session` per browser (up
   to 5 documents, each its own `IndexState`, plus the last 10 Q/A turns) in
   an in-memory dict, keyed by an `httponly` cookie; sessions expire after 2
   hours of inactivity and at most 50 are kept. Pipeline calls (embedding,
   LLM) run in a worker thread so one slow request doesn't block others.
   The last 3 turns are passed to the pipeline so follow-up questions
   resolve; answers must still come from retrieved passages.

2. **Frontend** (`static/index.html`, `static/style.css`, `static/app.js`)
   A single-page, mobile-first, vanilla HTML/CSS/JS UI: upload, document
   status bar, chat message list, per-answer sources disclosure. No
   framework, no build step. All dynamic content is inserted with
   `textContent`, never `innerHTML`, so document/model text can't inject
   markup.

3. **Document loader** (`pdf_loader.py`)
   Uses PyMuPDF (`fitz`) to open the PDF and extract text page by page.
   Output: a list of `(page_number, page_text)` pairs. Handles empty/invalid
   PDFs by returning an empty list (caught upstream as an error state).

4. **Chunker** (`chunker.py`)
   Splits each page's text into overlapping character-window chunks
   (fixed `chunk_size`, `chunk_overlap` — see "Chunking Configuration"
   below for why these aren't user-facing). Each chunk keeps a
   reference to its source page number. Output: list of
   `{"text": ..., "page": ...}` dicts.

5. **Embedder** (`embedder.py`)
   Wraps a `sentence-transformers` model (`all-MiniLM-L6-v2`) to turn chunk
   text and questions into vectors. The model is held in a module-level
   singleton so it loads once per process and is reused across requests —
   deliberately plain Python (not framework-specific caching), so the
   module stays usable from `main.py`, `evaluate.py`, and the test suite
   alike.

6. **Vector store / retriever** (`vector_store.py`)
   Wraps a FAISS `IndexFlatIP` (cosine similarity via normalized vectors).
   Builds the index from chunk embeddings; given a query embedding, returns
   the top-k most similar chunks with their page numbers and similarity
   scores.

7. **LLM client** (`llm_client.py`)
   Thin wrapper around one LLM API, selected and configured entirely via
   environment variables (`LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`) so the
   provider can be swapped without touching code. Builds a strict grounding
   prompt: system instruction + retrieved passages + question, with an
   explicit instruction to say the document doesn't contain the answer if
   the passages don't support one. Retrieved passages are fenced in
   delimiters and declared untrusted, so instructions embedded in a
   document are treated as quoted content rather than obeyed. Rate-limit
   (HTTP 429) responses are retried with backoff that honors `Retry-After`,
   bounded so a long back-off surfaces as an error instead of hanging.

8. **Orchestration** (`pipeline.py`)
   Ties the above together: `load → chunk → embed → index` at
   ingestion time, and `embed_query → retrieve → build_prompt → call_llm →
   return(answer, sources)` per question.

## Data Flow

```
PDF file
   │  PyMuPDF
   ▼
[(page_num, page_text), ...]
   │  chunker (size, overlap)
   ▼
[{text, page}, ...]  ──────────────► sentence-transformers ──► chunk embeddings
   │                                                                  │
   │                                                                  ▼
   │                                                         FAISS index (in memory)
   │
User question
   │  sentence-transformers
   ▼
question embedding ──► FAISS similarity search ──► top-k {text, page, score}
                                                          │
                                                          ▼
                                     prompt = system + top-k passages + question
                                                          │
                                                          ▼
                                                     LLM API call
                                                          │
                                                          ▼
                                        answer text  +  source list (page, passage)
                                                          │
                                                          ▼
                                      JSON response ──► app.js renders both
```

## RAG Pipeline (detail)

1. **Ingestion (once per uploaded document)**
   - Extract text per page (PyMuPDF).
   - Chunk each page's text with overlap so answers spanning a chunk
     boundary aren't lost.
   - Embed all chunks; build a FAISS index; keep a parallel Python list
     mapping index position → `{text, page}` for lookup after search.

2. **Query (once per question)**
   - Embed the question with the same model (required — mismatched models
     would produce meaningless similarity scores).
   - Retrieve top-k chunks by cosine similarity.
   - Construct a grounding prompt that includes the retrieved passages
     (each labeled with its page number) and instructs the LLM to answer
     **only** from them, and to say so explicitly if they don't contain the
     answer.
   - Call the LLM; return the answer text alongside the source chunks used,
     so the UI can display page/passage provenance independent of whether
     the LLM cites them correctly.

## Technology Choices

| Choice | Why |
|--------|-----|
| **FastAPI + vanilla HTML/CSS/JS** | A small, self-contained API plus a single static page is the simplest architecture that gives full control over mobile-responsive CSS — a full SPA framework (React/Vue/etc.) would mean a build step, `node_modules`, and far more moving parts than this project's single upload+chat page needs. Same origin for API and frontend means no CORS configuration either. See `DECISIONS.md` for the full comparison against staying on Streamlit and against a full SPA framework. |
| **PyMuPDF (fitz)** | Reliable page-level text extraction, handles most real-world PDFs, fast, no external binary dependency. |
| **sentence-transformers (`all-MiniLM-L6-v2`)** | Small (~80MB), fast on CPU, well-established baseline for semantic similarity — appropriate for a single-document tool at this scale. |
| **FAISS (`IndexFlatIP`)** | Exact (non-approximate) search is fine at this scale (one document, low thousands of chunks at most), simple to reason about, no external service to run. |
| **Configurable LLM API (env vars)** | NFR2 requires no hardcoded provider. A minimal `requests`-based OpenAI-compatible client is the default (works with OpenAI, Groq, or any OpenAI-compatible endpoint by changing `LLM_BASE_URL`), keeping the dependency footprint small. |
| **No LangChain/LangGraph** | Considered but deliberately not used — see `DECISIONS.md`. A hand-rolled pipeline better demonstrates pipeline understanding and keeps the code small and readable (NFR1), which a framework's abstractions would work against at this scale. |

## Chunking Configuration

`pipeline.py` defines `DEFAULT_CHUNK_SIZE`, `DEFAULT_CHUNK_OVERLAP`, and
`DEFAULT_TOP_K`, informed directly by the two-configuration evaluation in
`DECISIONS.md` (the defaults sit close to the configuration that performed
best there). The production UI does not expose these as user-facing
controls — deliberately, to keep the interface minimal and product-like
rather than exposing internal RAG hyperparameters to end users (that's
"developer demo" energy, not "polished product" energy). `pipeline.ingest`
and `pipeline.answer` still accept them as optional overrides, which is
exactly what FR8 (comparing two configurations) needs: `evaluate.py` calls
them directly with different values to reproduce the comparison against
any document, without needing UI controls at all.

## Failure Handling

- **Non-PDF / oversized upload**: rejected at the API boundary (`main.py`)
  with a 400/413 and a clear message, before the pipeline is ever invoked.
- **Invalid/empty PDF** (right type, but no extractable text): loader
  returns no pages → API responds with `{"error": ...}`, ingestion is
  blocked (no empty index built).
- **Unexpected exception during ingestion or answering** (e.g. the
  embedding model failing to load): caught generically in `main.py` so it
  degrades to a clean error response instead of an unhandled 500.
- **Missing API key**: LLM client raises a clear configuration error on
  first call, caught by the API and returned as `{"error": ...}`, rendered
  by the frontend as an error-styled chat message — not a stack trace.
- **Rate limiting / API failure**: retried with bounded backoff, then
  surfaced as a readable error in the conversation. `evaluate.py`
  additionally paces its requests and records a per-question failure
  rather than discarding a whole batch run.
- **No relevant chunks found** (e.g. off-topic question): retrieval still
  returns its top-k (FAISS always returns *something*), but similarity
  scores will be low; the prompt's grounding instruction handles the
  "answer not in document" case at the LLM level — see `DECISIONS.md` for
  the actual evaluation of this behavior against a real document.
