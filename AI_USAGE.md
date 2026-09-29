# AI Usage — AI Document Assistant

Honest, running record of how AI assistance (Claude Code) was used while
building this project. Updated as work progresses — not written once at
the end.

## Planning Phase

- Claude Code was used to draft the initial planning documents
  (`PROJECT_SPEC.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`,
  `DECISIONS.md`, this file) from a set of functional requirements the
  user defined up front (single-document PDF QA, page-aware chunking,
  grounded retrieval-augmented answers, source citations, and a
  reproducible two-configuration chunking/retrieval comparison).
- The user directed the workflow (planning before coding, phase
  structure, what stack to use — all specified up front) and reviewed
  each planning document before implementation began.
- Architectural choices (e.g. not using LangChain, model/library
  selection) were proposed by Claude Code with stated reasoning, for the
  user to accept, reject, or change before implementation began.

## Implementation Phase

- All source files (`pdf_loader.py`, `chunker.py`, `embedder.py`,
  `vector_store.py`, `llm_client.py`, `pipeline.py`, `evaluate.py`, and
  originally a Streamlit `app.py`, later replaced by `main.py` + `static/`
  — see "UI/Architecture Migration" below) were drafted by Claude Code
  directly from the approved
  `PROJECT_SPEC.md`/`ARCHITECTURE.md`/`IMPLEMENTATION_PLAN.md`, then
  actually run and tested (see `DECISIONS.md`'s development log) rather
  than assumed to work.
- The user made two explicit implementation-affecting decisions during
  this phase: (1) keep chunk_size/chunk_overlap/top_k configurable in the
  pipeline (initially via an "Advanced settings" expander in the
  Streamlit UI; after the FastAPI migration they are no longer in the UI
  and are used only by `evaluate.py`); (2) require a dedicated
  `evaluate.py` script for the two-config comparison instead of relying
  only on manual UI clicking, so the comparison is reproducible.
- The user supplied real API credentials (OpenAI, then Groq after the
  OpenAI key turned out to have no billing credits) directly in chat for
  Claude Code to place into a local, gitignored `.env` file. Keys were
  never printed back, hardcoded into source, or committed.
- No AI-suggested approach has been rejected; the OpenAI→Groq switch was
  caused by an account billing issue, not a design change.

## Testing Phase

- Claude Code selected and downloaded a real public PDF (a well-known
  arXiv paper) to validate the pipeline on realistic document structure
  beyond the bundled minimal demo PDF, read its actual extracted content
  before writing test questions (rather than guessing), and ran a real
  two-configuration comparison via `evaluate.py`. All answers, sources,
  and chunk/config counts recorded in `DECISIONS.md` are the actual
  script output from that run — none were written from assumption.
- Mid-run, the evaluation hit a genuine Groq API rate limit (429, TPM
  exceeded). This was diagnosed with a direct API probe before being
  treated as a bug, then fixed with retry/backoff logic in
  `llm_client.py`. The fix and the reasoning behind it are logged in
  `DECISIONS.md`'s development log, not hidden.
- A later console warning (`ModuleNotFoundError: No module named
  'torchvision'`) was diagnosed the same way: reproduced first, root
  cause traced to Streamlit's own file-watcher code (not this project's
  code), and fixed with a targeted Streamlit config change rather than
  installing an unused dependency to mask the symptom.

## Generalization Pass

- The project was later converted into a standalone, general-purpose
  portfolio project. Claude Code performed a repository-wide audit for
  organization- and context-specific references and removed or reworded
  them across the UI, README, planning docs, and code comments, while
  explicitly preserving every technically meaningful decision and finding
  (the chunking/retrieval comparison results, the rate-limit
  retry/backoff fix, the file-watcher fix) rather than deleting them.
- No RAG architecture, pipeline behavior, or dependency was changed as
  part of this pass — it was a documentation/presentation change,
  verified by re-running the full test suite afterward (see
  `DECISIONS.md`).

## Review and Hardening Pass

- Claude Code ran a project-wide review (architecture, RAG correctness,
  security, PDF handling, state management, UI/UX, deployment,
  dependencies, documentation) and fixed the issues it found. The full
  list, including what was actually broken and how each fix was
  verified, is in `DECISIONS.md`.
- Findings were verified rather than assumed: the `Retry-After` parsing
  fix was unit-tested across nine header forms, the prompt-injection
  defense was tested against a purpose-built malicious PDF, and the
  error-state fix was confirmed in a real browser.
- After hardening the system prompt, the full two-configuration
  evaluation was re-run to confirm the previously documented findings
  were still accurate rather than silently invalidated.

## Autonomous Maintenance Pass

- Claude Code was authorized to act as an autonomous maintainer: inspect
  the repository, fix genuine issues on its own judgment, and document
  what changed, without asking permission for each individual fix.
- It re-read every source file directly rather than relying on earlier
  context, and found three real, previously-unnoticed gaps (unhandled
  exceptions on the ingestion/query paths, a missing-API-key crash in
  `evaluate.py`, a stale README placeholder) — all fixed and tested; see
  `DECISIONS.md` for specifics.
- It also identified that all prior test verification in this project
  existed only as throwaway scripts in a temp directory, never committed
  — a real gap given `CONTRIBUTING.md` itself calls out the missing test
  suite as desirable — and added a genuine, committed `pytest` suite plus
  a credential-free CI workflow to close it, rather than leaving that
  gap unaddressed because it wasn't explicitly requested this time.
- While verifying the new test suite, it caught its own mistake — a
  fixture that silently skipped every LLM-dependent test even with a
  valid key configured, because `.env` wasn't being loaded — rather than
  reporting the misleading "6 skipped" result as if it were success.

## UI/Architecture Migration

- Claude Code was authorized to independently choose and implement a
  replacement for the Streamlit UI, whose mobile experience was the
  motivating problem. It evaluated staying on Streamlit, a full SPA
  framework, and a minimal FastAPI + static-frontend approach, and chose
  the last one; the reasoning is in `DECISIONS.md`, not just the outcome.
- It deliberately left the entire RAG pipeline (`pdf_loader.py` through
  `pipeline.py`) untouched, confirming this was possible before starting
  by checking that `evaluate.py` and the `tests/` suite only import the
  pipeline modules, never the UI layer — so the migration couldn't
  silently change retrieval or grounding behavior.
- It found and fixed three real bugs in its own new code before calling
  the work done: a CSS specificity bug that would have broken the
  document-view's hide/show logic, a chat input that didn't stay pinned
  to the screen bottom with few messages, and a devcontainer config
  (added by the user in an earlier, unrelated commit) that would have
  launched a command referencing a file this migration deleted. None of
  these were requested explicitly — they were caught by actually running
  and looking at the result, not assumed to be fine because the code
  looked reasonable.
- A visual "this doesn't look centered" impression from a screenshot was
  checked against actual computed CSS values before being treated as a
  bug — it wasn't one (the layout was already correct; the impression
  came from a small element in a wide, empty viewport). This is recorded
  so a "found and fixed N issues" summary doesn't imply every impression
  during review turned out to be a real problem.

## Live Deployment

- Claude Code evaluated multiple hosting platforms against real,
  researched or empirically-tested constraints — not assumptions. Render
  was tested by actually running the built image with `--memory=512m`
  and measuring 99.4% usage at idle before rejecting it; Fly.io/Railway
  were rejected based on their actual current (2026) free-tier terms,
  researched rather than recalled from training data after already being
  wrong once about Hugging Face Spaces' pricing in this same session.
- It correctly stopped and asked for account-level action, rather than
  attempting workarounds, at two genuine credential boundaries: it never
  asked for AWS secret access keys after being told not to (accepting
  scoped SSH-only access to a single instance instead), and it never
  attempted to create AWS/Oracle/GCP accounts itself, since those require
  identity/card verification only the user could complete.
- It found and fixed a real deployment bug (PyPI's default Linux torch
  wheel pulling unnecessary multi-hundred-MB NVIDIA CUDA packages) by
  reading the actual build log during deployment, not by inspecting the
  Dockerfile in the abstract — the problem was only visible while it was
  happening.
- It caught and disclosed its own mistake (an overly broad `docker
  system prune -af` deleting the image it had just built) in the same
  turn it happened, rather than only in retrospect.
- When SSH access to the deployed instance stopped working partway
  through final verification, it did not fabricate or extrapolate a live
  memory reading. It reported the actual local measurement it had (from
  running the identical image under the same load) as exactly that — a
  local measurement, not a live one — and flagged the SSH loss as an
  open item for the user to check, rather than guessing at a cause.

## Dependency Maintenance

- Five routine Dependabot version-floor bumps (`pytest`, `requests`,
  `pymupdf`, `uvicorn`, `faiss-cpu`) were reviewed and merged only after
  CI passed on each; one had a `requirements.txt` merge conflict that
  Dependabot's own rebase resolved. No source code changed, so no
  redeploy was needed. Released as `v1.1.2`. Details are in
  `DECISIONS.md` (2026-09-22).

## What the author must be able to explain

AI wrote most of the code here, so this lists the parts that must be
understood and explainable without AI help (the task requires it):

- **Chunking** (`chunker.py`): character sliding window per page with
  overlap, why chunk size mattered more than top-k in the A/B evaluation.
- **Embedding + retrieval** (`embedder.py`, `vector_store.py`): MiniLM
  vectors are normalized, so FAISS inner product equals cosine similarity;
  `IndexFlatIP` is exact search.
- **Grounding** (`llm_client.py`): the system prompt restricts answers to
  fenced passages and defines the exact "not found" refusal string.
- **Why no LangChain**: see `DECISIONS.md` decision #1 and its outcome.

## Principles followed

- No fabricated test results, decisions, or requirements — ever.
- Every AI-authored planning document is explicitly reviewed against the
  project's actual requirements before implementation proceeds.
- This file is updated alongside the work, not reconstructed after the
  fact.

## Bug-Fix and Bonus-Feature Pass (2026-09-29)

- The user asked Claude Code to fix any errors, close gaps against the task
  brief, and add improvements of its choice. It compared the repository to
  the brief first, then read every source file for defects even though all
  tests passed.
- It found and fixed real bugs (blocking calls on the event loop, a 500 on
  invalid JSON, errors sent with HTTP 200, unbounded session memory, a
  frontend that froze on non-JSON proxy errors). Each is listed in
  `DECISIONS.md` with what went wrong.
- It implemented the brief's three bonus features (multiple documents,
  conversation history, document summary) and chose the simpler option at
  each step with the rejected alternative recorded (per-document indexes
  vs. one merged index; query concatenation vs. LLM query rewriting;
  sampled vs. map-reduce summary). Rate limits drove the last two choices.
- It kept the single-document path identical and added tests proving it,
  so the earlier Config A/B evaluation still holds without a re-run.
- It checked a suspicious result before reporting it: a follow-up about
  Jupiter's moons was refused, and the document text showed Jupiter's moons
  are never mentioned, so the refusal was correct. It then re-tested
  follow-ups with a question the document can answer (Saturn).
- It did not re-run the evaluation on an organizer-provided document,
  since none is in the repository; that item stays open.
