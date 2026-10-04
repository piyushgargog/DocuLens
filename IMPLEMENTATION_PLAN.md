# Implementation Plan — DocuLens

Each phase should be independently runnable/testable before moving to the
next. Phases 0–7 deliver all core requirements; 8–9 are testing and docs;
optional enhancements (Phase 10) only happen if 0–9 are solid.

> **This is the original plan, kept as a record of how the project was
> built.** Each phase ends with a _Status_ line describing what exists now
> (v1.4.1). In short: the pipeline (Phases 1–6) and the testing approach
> (Phase 8) were followed as written; the UI (Phase 7) was rebuilt several
> times — Streamlit form → Streamlit chat → FastAPI + static frontend →
> two redesigns (`DESIGN.md`); all Phase 10 enhancements were built. The
> "After the plan" section at the end lists what was added beyond it.
> References to `app.py`/`streamlit run` below are historical.

## Phase 0 — Project Setup
- `requirements.txt` (originally streamlit, pymupdf, sentence-transformers, faiss-cpu, requests, python-dotenv, numpy — streamlit was later replaced by fastapi/uvicorn/python-multipart, see `DECISIONS.md`).
- `.env.example` documenting `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`.
- Folder skeleton: `app.py`, `pdf_loader.py`, `chunker.py`, `embedder.py`, `vector_store.py`, `llm_client.py`, `pipeline.py`, `tests/` (manual test notes), `sample_docs/`.
- **Test**: `pip install -r requirements.txt` succeeds; `streamlit run app.py` opens a blank page without errors.
- **Depends on**: nothing.
- _Status:_ done; today the entry point is `main.py` (`uvicorn main:app`), dev tools live in `requirements-dev.txt`, and `tests/` holds the `pytest` suite.

## Phase 1 — Document Loader (`pdf_loader.py`)
- Function: open PDF bytes/path with PyMuPDF, return `[(page_num, text), ...]`.
- Handle empty PDF / zero extractable text as an explicit empty result, not an exception leak.
- **Test**: run against a real multi-page PDF and confirm page count + spot-checked text; run against an empty/corrupt PDF and confirm a clean empty result rather than a crash.
- **Depends on**: Phase 0.
- _Status:_ as planned; the document is opened with a context manager so a mid-extraction failure can't leak the handle.

## Phase 2 — Chunker (`chunker.py`)
- Function: given `[(page_num, text), ...]`, `chunk_size`, `chunk_overlap` → `[{"text", "page"}, ...]`.
- Character-based sliding window per page (simple, predictable, easy to explain — appropriate for a small, single-document RAG tool).
- **Test**: unit-check chunk count and overlap behavior on a known string; confirm every chunk carries the correct page number.
- **Depends on**: Phase 1 (consumes its output format).
- _Status:_ as planned; chunks are additionally tagged with the document's filename (`doc`) for multi-document sources.

## Phase 3 — Embedder (`embedder.py`)
- Load `all-MiniLM-L6-v2` once (cached via `st.cache_resource` in the app layer).
- Function: `embed(texts: list[str]) -> np.ndarray`.
- **Test**: embed a few sample sentences, confirm output shape `(n, 384)` and that similar sentences score higher cosine similarity than dissimilar ones (quick manual sanity check).
- **Depends on**: Phase 0 only (independent of chunker).
- _Status:_ built, but caching is a module-level singleton in `embedder.py`, not `st.cache_resource`, so it works outside any web framework.

## Phase 4 — Vector Store (`vector_store.py`)
- Build a FAISS `IndexFlatIP` from normalized chunk embeddings.
- Function: `search(query_embedding, top_k) -> [(chunk_index, score), ...]`, resolved back to `{text, page, score}` via a parallel metadata list.
- **Test**: build an index from a handful of known chunks, run a query whose answer is obviously in one chunk, confirm that chunk ranks first.
- **Depends on**: Phase 2 (chunk format) + Phase 3 (embeddings).
- _Status:_ as planned; one index per document, and `search()` returns every chunk key (incl. `doc`) with the score.

## Phase 5 — LLM Client (`llm_client.py`)
- Minimal OpenAI-compatible chat completion call via `requests`, config from env vars.
- Grounding prompt template: system instruction ("answer only from the provided passages; if they don't contain the answer, say so explicitly") + numbered passages with page labels + question.
- Raise a clear, caught-upstream error if `LLM_API_KEY` is unset.
- **Test**: call with a small fixed context + question with a known answer; call with context that doesn't contain the answer and confirm the model says so; call with the API key unset and confirm a clean error, not a stack trace to the user.
- **Depends on**: Phase 0 only.
- _Status:_ built and hardened: passages fenced as untrusted content (prompt-injection defense), 429 retry honoring `Retry-After` (capped at 30s), optional conversation history, separate summary prompt.

## Phase 6 — Pipeline (`pipeline.py`)
- `ingest(pdf_bytes, chunk_size, chunk_overlap) -> index_state`
- `answer(question, index_state, top_k) -> {answer, sources: [{page, text, score}]}`
- Wires Phases 1–5 together.
- **Test**: end-to-end call against a real PDF with a known-answerable question; confirm sources returned match where the answer actually is.
- **Depends on**: Phases 1–5.
- _Status:_ built; now also `retrieve()` (multi-document, follow-up aware) and `summarize()`.

## Phase 7 — Streamlit UI (`app.py`)
> Note: this describes the original plan. The UI was later rebuilt as a
> chat interface (message history, per-answer sources popover, sidebar
> document card) — see `DECISIONS.md` for that change and its rationale.
- Upload widget → calls `pipeline.ingest`.
- Sidebar: chunk size, chunk overlap, top-k sliders (defaults + ability to change and re-ingest).
- Question input → calls `pipeline.answer` → renders answer, then an expandable "Sources" section per result showing page number + passage text (+ similarity score).
- Error states surfaced as `st.error(...)`, not raw tracebacks.
- **Test**: manual click-through in browser — upload, ask, see answer + sources; trigger each error state (empty PDF, missing key) and confirm graceful messages.
- **Depends on**: Phase 6.
- _Status:_ replaced. The UI is now `main.py` (FastAPI) + `static/` (vanilla HTML/CSS/JS). Hyperparameter sliders were removed from the UI; `evaluate.py` covers FR8.

## Phase 8 — Testing
- Run ≥5 questions against a real document, recorded with actual answers/sources (not fabricated).
- Include ≥1 question intentionally unanswerable from the document; confirm the assistant declines rather than hallucinates.
- Run the same question set under **two** chunking/retrieval configurations (e.g. small chunks/low top-k vs. larger chunks/higher top-k); record observed differences in `DECISIONS.md`.
- Also exercise: invalid/empty PDF, missing API key.
- **Depends on**: Phase 7 (needs the full app running).
- _Status:_ done (see `DECISIONS.md`, Real-World Validation); plus a committed `pytest` suite (153 tests) and CI.

## Phase 9 — README + Final Review
- Write `README.md`: overview, architecture summary, setup, how to run, usage, testing/results, limitations.
- Cross-check every mandatory requirement in `PROJECT_SPEC.md` against the actual running app.
- **Depends on**: Phase 8 (results must be real before documenting them).
- _Status:_ done; README, `ARCHITECTURE.md` and this file are kept current with each release.

## Phase 10 — Optional Enhancements (only if 0–9 are complete and solid)
- Multiple documents (extend ingestion to merge multiple sources, tag chunks with doc name + page).
- Conversation history (chat-style message list in Streamlit session state, prior turns optionally included in prompt).
- Short document summary (one extra LLM call over a sample of chunks or full text if small enough).
- _Status:_ all three built in v1.2.0 — up to 5 documents, last-3-turn follow-ups (history kept server-side, not in Streamlit), sampled 10-chunk summary.

## Testing Strategy Summary
- Each phase (1–6) got a quick manual/functional check before moving on. (Those ad-hoc checks were later turned into the committed `pytest` suite in `tests/`.)
- Phase 8 is the authoritative test pass against this project's own requirements (`PROJECT_SPEC.md`) and is what gets written up in `DECISIONS.md`/`README.md`.
- No test result is recorded anywhere unless it was actually observed from a real run.

## After the plan
Added later, each with its reasoning in `DECISIONS.md`:
- FastAPI backend with bounded in-memory sessions, worker-thread pipeline
  calls, real HTTP status codes, and `Cache-Control: no-cache` on the
  frontend.
- Security hardening: prompt-injection defense, `httponly`/`samesite`/
  `Secure` session cookie, no exception text returned to clients, upload
  and question size limits.
- Clickable page citations and sources beside each answer, light/dark
  themes, keyboard-accessible upload.
- Measured retrieval evaluation (`retrieval_eval.py`): Hit@k and MRR on a
  labelled set, BM25 baseline vs two embedding models vs hybrid.
- LLM beyond extraction (v1.7.0): whole-document question routing,
  explain-in-own-words prompt with partial answers, suggested questions,
  "Explain more simply" / "Go deeper", fallback model on rate limits.
- v2.0.0: streamed answers (SSE) with stop, hybrid BM25 + embedding
  retrieval in production, per-question document scope, copy/export,
  new composer, concurrency limits, CSRF guard, `__Host-` cookie,
  `pip-audit` in CI.
- v3.6.2: the landing settle-in and footer heartbeat keep their original quick
  timing under "reduce motion" (v3.6.1's slower fades felt wrong).
- v3.6.1: fixes after the first real-document test: suggested questions
  appear once (placeholders, no flash), the landing fade and footer heart work
  with "reduce motion" on, and the v3.6.0 similarity floor is off by default
  after it refused answerable resume questions; "main focus / purpose of this
  document" is a whole-document question.
- v3.6.0: production-hardening pass — `TRUSTED_PROXIES` for correct per-IP
  rate limiting behind Docker/Nginx, an optional retrieval-score abstention
  floor (`score_floor_eval.py`; off by default since v3.6.1), prompt-fence/filename/history injection
  hardening, per-session locks, a server-wide `MAX_TOTAL_CHUNKS` budget, a
  sliding session-cookie expiry, mid-stream failure handling, and optional
  startup model prefetch.
- v3.0.0–v3.1.1: rebrand to DocuLens (new repo + domain), structured
  logging with per-request IDs and optional Sentry (`observability.py`).
- v3.2.0: premium Markdown answer rendering (safe, no `innerHTML`).
- v3.3.0–v3.4.0: redesign to "Calm Light" (Claude-style: ink + warm white +
  terracotta accent, Newsreader serif + Inter); fixed a frozen indexing
  spinner under `prefers-reduced-motion`.
- v3.5.0: UX polish — reading-screen facts and progress bar, instant
  "Try asking" starters, thinking-panel polish, catchier landing page.
- v2.3.0: "show thinking" (streamed model reasoning in a collapsible panel
  via a `reasoning` SSE event), OCR for scanned PDFs (Tesseract, bounded),
  a nicer loading/streaming animation, and a compact mobile footer.
- v2.2.0: several AI providers with failover and cooldowns
  (`providers.py`: Groq, OpenRouter, NVIDIA, Hugging Face; Google AI
  Studio opt-in), "answered by"
  on every answer, `/api/status` + live provider status in a two-tier
  footer, `/api/health` + Docker HEALTHCHECK, light/dark toggle, toasts,
  jump-to-latest, smooth scrolling, loading skeleton.
- v2.1.0: visible redesign ("Modern product", `DESIGN.md`): Inter,
  sidebar and conversation cards, question bubbles, source cards, floating
  composer, full footer with credit, links, privacy dialog and version.
- Document retention: uploads are never written to disk; extracted text is
  deleted on removal or after 2 hours idle, enforced by a background task.
- `pytest` suite (127 tests) with GitHub Actions CI, CodeQL and Dependabot.
- Docker image (CPU-only PyTorch) deployed to AWS EC2 behind Nginx with
  HTTPS.
