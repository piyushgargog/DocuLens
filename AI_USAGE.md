# AI Usage — DocuLens

Honest, running record of how an AI coding assistant was used while
building this project. Updated as work progresses — not written once at
the end.

## Planning Phase

- The AI coding assistant was used to draft the initial planning documents
  (`PROJECT_SPEC.md`, `ARCHITECTURE.md`, `IMPLEMENTATION_PLAN.md`,
  `DECISIONS.md`, this file) from a set of functional requirements the
  user defined up front (single-document PDF QA, page-aware chunking,
  grounded retrieval-augmented answers, source citations, and a
  reproducible two-configuration chunking/retrieval comparison).
- The user directed the workflow (planning before coding, phase
  structure, what stack to use — all specified up front) and reviewed
  each planning document before implementation began.
- Architectural choices (e.g. not using LangChain, model/library
  selection) were proposed by the AI coding assistant with stated reasoning, for the
  user to accept, reject, or change before implementation began.

## Implementation Phase

- All source files (`pdf_loader.py`, `chunker.py`, `embedder.py`,
  `vector_store.py`, `llm_client.py`, `pipeline.py`, `evaluate.py`, and
  originally a Streamlit `app.py`, later replaced by `main.py` + `static/`
  — see "UI/Architecture Migration" below) were drafted by the AI coding assistant
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
  the AI coding assistant to place into a local, gitignored `.env` file. Keys were
  never printed back, hardcoded into source, or committed.
- No AI-suggested approach has been rejected; the OpenAI→Groq switch was
  caused by an account billing issue, not a design change.

## Testing Phase

- The AI coding assistant selected and downloaded a real public PDF (a well-known
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
  portfolio project. The AI coding assistant performed a repository-wide audit for
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

- The AI coding assistant ran a project-wide review (architecture, RAG correctness,
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

- The AI coding assistant was authorized to act as an autonomous maintainer: inspect
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

- The AI coding assistant was authorized to independently choose and implement a
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

- The AI coding assistant evaluated multiple hosting platforms against real,
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
- **Multiple documents and follow-ups** (`pipeline.retrieve()`/`answer()`):
  one index per document, merged by score; follow-up retrieval also uses
  the previous question; earlier turns may resolve references but are
  never evidence.
- **Document retention** (`main.py`): PDFs are never written to disk; text
  lives in the in-memory session until removal or 2 hours idle, enforced by
  a background task.
- **Frontend safety** (`static/app.js`): all document/model text is inserted
  as text, never HTML; citations are built with DOM nodes.
- **Why no LangChain**: see `DECISIONS.md` decision #1 and its outcome.

## Principles followed

- No fabricated test results, decisions, or requirements — ever.
- Every AI-authored planning document is explicitly reviewed against the
  project's actual requirements before implementation proceeds.
- This file is updated alongside the work, not reconstructed after the
  fact.

## Bug-Fix and Bonus-Feature Pass (2026-09-29)

- The user asked the AI coding assistant to fix any errors, close gaps against the task
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

## Frontend Direction Pass (v1.4.0)

- The user installed an open-source design-direction skill
  (jangles-byte/atelier, MIT) and asked for the frontend to be improved
  with it. The AI coding assistant read the skill's files before
  installing it, then followed its method: audit the existing UI against
  the skill's anti-generic checklist, write a design philosophy
  (`DESIGN.md`) before changing code, then render and critique.
- The critique pass found and fixed layout issues from screenshots, and a
  real accessibility bug by keyboard-testing: the upload area had never
  been reachable with the Tab key. Details are in `DECISIONS.md`.

## Review, Documentation and Privacy Passes (v1.4.1–v1.4.2)

- **v1.4.1:** the user asked why a fixed Saturn example appeared on the
  upload screen. The AI coding assistant had added it as a decorative
  specimen without labelling it; it acknowledged that this contradicted
  the project's no-hardcoded-content rule, removed it, and searched the
  application source to confirm no sample-document content remained.
- **Documentation refresh:** every document was re-read against the code.
  `ARCHITECTURE.md` was rewritten and the spec, plan, README, security and
  contribution docs were corrected where they still described the
  single-document app. Chronological logs (this file, `DECISIONS.md`) were
  left as written.
- **v1.4.2:** the user reported that answers auto-scrolled to the bottom
  and asked whether uploaded PDFs are deleted. The assistant checked what
  is actually stored before changing anything (nothing on disk; text in
  memory), found a real gap (expiry only ran on requests) and fixed it with
  a background sweep, and — while measuring the scroll fix — found a mobile
  layout bug that pushed the ask box off-screen. Each fix has a test or a
  recorded browser measurement.
- Throughout, releases were only tagged after CI passed and the live site
  was checked in a real browser.

## Measured Retrieval Evaluation

- The user noted that the other AI/ML tasks in the brief looked heavier.
  The AI coding assistant compared the four tasks' requirements and
  concluded the gap was measurement, not features: the others report
  metrics against a baseline, while this project's evaluation was
  qualitative. With the user's agreement it added `retrieval_eval.py`
  (Hit@k, MRR, a BM25 baseline, two Hugging Face embedding models and a
  hybrid) without changing the app.
- The labelled set was built from the extracted text of each page, with
  every label's evidence phrase checked automatically against its page.
- The result that the keyword baseline beat the app's own embedding model
  was reported as found, with its likely bias (questions written from the
  document's wording) stated rather than hidden.
- While labelling, the assistant found that an earlier claim in this
  project's own logs was wrong: the "misread BLEU 41.0" is actually printed
  on page 8 of the paper. The claim was corrected where it appeared, with
  the original log entries kept and annotated.

## Credit, Version and Motion (v1.6.0)

- At the user's request the AI coding assistant added an author credit,
  repository link and version to the UI, plus animation and polish. It
  kept the motion inside the existing design direction (only in response
  to an action, off under reduced motion) and recorded the credit's heart
  as a deliberate exception in `DESIGN.md`, rather than quietly breaking
  the "yellow means evidence" rule.
- It made the version a single source of truth with a test, since an
  earlier release had shipped a stale-cache bug caused by a forgotten
  version bump.

## Making the LLM's Role Real (v1.7.0)

- The user said the assistant felt like a text extractor. Before changing
  anything, the AI coding assistant ran real questions and traced the
  weak answers to causes (retrieval returning the reference list for
  whole-document questions; a prompt that asked for brevity and all-or-
  nothing answers), rather than guessing.
- Its own testing surfaced a grounding failure it had just introduced
  ("Go deeper" padded a short document with outside knowledge); it
  reported and fixed it, and re-tested.
- Its test runs used up the LLM provider's daily quota — which the live
  site shares. It said so and added a fallback model, and it labelled which
  model produced each result instead of presenting fallback-model output as
  if it came from the main model.

## Security Audit (v1.8.0)

- The user supplied a five-part security checklist (a PDF of audit
  prompts). The AI coding assistant read it fully and treated it as a list
  of things to verify — against the code, every commit in git history, and
  the live deployment — rather than as instructions to trust.
- It confirmed issues live before fixing them (public API docs, missing
  headers, the app served over plain HTTP on the raw IP) and marked the
  checklist items that don't apply (no passwords, payments or database).
- It added a disclosure it had previously missed: its own earlier privacy
  note said the PDF isn't saved, but not that passages are sent to the LLM
  provider.

## Version 2.0.0

- The user asked for a genuine v2. The AI coding assistant proposed the
  architecture (streaming, hybrid retrieval from the earlier measurement,
  load control), security and UX changes, then built them with tests and
  browser checks.
- It verified that moving BM25/RRF into a shared module changed no
  measured number, and that the app now scores exactly what the
  evaluation measured, before claiming the improvement.
- Its own screenshots exposed a bug it had introduced (streamed text
  decoded as ISO-8859-1); it traced the cause, fixed it and added a
  regression test.
- It reported that half of the answer-level re-run came from the fallback
  model because the main model's quota ran out, rather than presenting the
  results as the main model's.

## Version 2.1.0

- The user said the earlier redesign hadn't visibly improved the UI or the
  footer. The AI coding assistant offered three concrete directions with
  mock-ups. The user chose "Modern product", and the assistant rewrote the
  stylesheet, the markup and `DESIGN.md` to match.
- It kept every element ID the JavaScript depends on, so behaviour and
  tests were unchanged. It checked the result with headless-browser
  screenshots (desktop light and dark, and a 390px phone). Those
  screenshots showed the start page running under the footer on a phone,
  which it fixed before release.

## Version 2.2.0

- The user asked for more than one AI provider ("one alone won't do"),
  naming Groq, Hugging Face, OpenRouter and NVIDIA, and later asked for
  Google AI Studio to be tried first. The assistant added it, but in testing
  Gemini's free tier was overloaded (503) or rate limited on most calls, so
  its injection safety couldn't be verified. The user then asked to ship
  what works, and Gemini was made opt-in. The AI coding assistant
  noticed that all five offer OpenAI-compatible APIs. It built a failover chain with per-provider
  cooldowns instead of four client integrations, and kept older `.env`
  files working.
- It wrote unit tests with fake HTTP responses for each failure path (rate
  limit, bad key, timeout, 5xx, mid-stream failure) before verifying against
  the real providers.
- It also added the visible pieces the user asked for (a smoother UI and a
  better footer): provider status in the footer, "answered by" on answers,
  a theme toggle, toasts and jump-to-latest. It checked them in a headless
  browser, which caught toasts covering the footer.

## Version 2.3.0

- The user asked for five things: a cleaner mobile footer, OCR for scanned
  PDFs, fixes to broken UI, a nicer "loading/answer" animation, and a
  ChatGPT-style "show thinking" feature.
- The assistant checked the providers first and found Groq's gpt-oss and
  OpenRouter's Nemotron both stream a `reasoning` field, so "show thinking"
  is real reasoning, not a fake spinner. It buffers reasoning from a route
  until that route commits to an answer, so a failed provider's thoughts
  never leak.
- OCR was added as a bounded fallback (only for PDFs with no text layer, at
  most 30 pages) and verified on the server before shipping.
- It reproduced the mobile footer and the broken thinking-panel collapse
  from screenshots and fixed both.

## Version 2.4.0

- The user said the footer was still "fixed" and the UI looked "AI type"
  (generic). The assistant loaded the design-direction skill, checked the
  current look against the anti-generic checklist, and found it *was* the
  default-AI template (Inter + indigo + soft shadows + uniform fade-ups).
- It committed to a named direction ("Reading Room") in DESIGN.md before
  changing a pixel: warm paper, a self-hosted Fraunces editorial serif for
  headlines, Inter for the UI, mono numerals, one ink-green accent, and amber
  reserved only for citations.
- The footer was made a real footer — an in-flow colophon on the opening page
  (reached by scrolling), hidden in the conversation, with version and
  provider health moved to the top bar.
- Rendered and critiqued in a headless browser; caught and fixed the mobile
  top bar wrapping the wordmark to three lines.

## Version 3.0.0

- The user rebranded the project to **DocuLens**. The assistant renamed the
  GitHub repo, updated the git remote, rebranded every user-facing string and
  in-repo link (repo + domain), renamed the Docker image/container, moved the
  live site to doculens.duckdns.org, and renamed the local folder — verifying
  the test suite stayed green throughout.

## Version 3.1.0

- Added observability (structured logging, per-request IDs, optional Sentry)
  as the production-hardening step, keeping user document content out of logs
  and error reports. Re-tested the live site from a Kali box with nmap/nikto/
  sslscan and found no vulnerabilities.

## Versions 3.2.0 – 3.5.0

- **3.2.0:** the user wanted answers to read like Claude/ChatGPT rather than
  flat text. The assistant wrote a strictly safe Markdown renderer (every node
  built with `createElement`/`textContent`, never `innerHTML`) so untrusted
  model or document text still cannot inject markup, and verified it headless
  in light and dark.
- **3.3.0:** the user found the "Reading Room" look fancy rather than serious
  and noticed a frozen indexing spinner. The assistant moved to a neutral
  Inter-only look and fixed the spinner (a `prefers-reduced-motion` rule had
  stopped its animation).
- **3.4.0:** the user supplied Claude's exported design tokens as the
  reference. The assistant applied them via the frontend-design skill — ink
  and warm near-white, one terracotta accent, Newsreader serif over Inter —
  and checked light and dark in a headless browser.
- **3.5.0:** UX polish (reading-screen facts and progress bar, instant
  starter questions, thinking-panel polish, a landing page with an example
  cited answer). Frontend only; verified headless with zero console errors.

## Version 3.6.0

- Production hardening, driven by two outside code reviews of the repository
  (their findings were treated as claims to verify, not instructions). The
  assistant read both proposed pull requests line by line before touching
  anything. Of the proposed changes it kept the ones that held up — trusted-proxy
  CIDR handling for rate limiting, neutralising prompt-fence tokens in uploaded
  text and filenames, labelling earlier chat turns as untrusted, per-session
  locks, a server-wide chunk budget, wrapping mid-stream network failures — and
  rejected or reworked the rest: one proposal removed the "show thinking"
  feature while claiming no breaking changes; another shipped a "prefetch" that
  recursed forever when enabled (reproduced with a stub model) and a similarity
  floor with no measurement behind it.
- The retrieval abstention floor was measured instead of guessed
  (`score_floor_eval.py`): the assistant wrote the script, ran it on both
  sample documents with answerable, off-topic and cross-document questions, and
  chose 0.25 over the proposed 0.30 because 0.30 wrongly refused 8% of
  answerable questions against 3%.
- Tests were rewritten so every new behaviour is covered (the proposals shipped
  with almost none). 146 tests pass.

## Version 3.6.1

- Fixes after the user's first real-document test. The assistant had chosen and
  documented a retrieval-similarity floor in 3.6.0 from two sample documents; the
  user's own resume exposed that it refused answerable questions (including one
  the app itself suggested). The assistant reproduced it by scoring the resume's
  questions, concluded similarity cannot separate short or pronoun-heavy
  questions from off-topic ones, turned the floor off by default, rewrote the
  docs to say so plainly, and added "test with a short real document" to the
  contributor guide.
- It also routed "main focus / purpose of this resume" questions as
  whole-document questions, made the starter questions appear once, and found
  that the "dead" footer heart and pop-in landing page came from the OS
  reduced-motion setting rather than a missing animation, then verified the
  result in a headless browser under both motion settings. 153 tests pass.

## Version 3.6.2

- The user said the v3.6.1 motion fix felt slow and the heart faded instead of
  beating. The assistant had chosen softer, slower fades as a "safe" reading of the
  reduced-motion setting without asking; it reverted to the original quick timing
  for the page settle-in and the heartbeat (keeping everything else reduced) and
  re-verified in a headless browser.
