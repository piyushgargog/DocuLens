# Contributing to DocuLens

Thanks for your interest in this project. It's a personal/portfolio project
maintained by Piyush Garg in spare time,
so please set expectations accordingly — issues and PRs are welcome, but
there's no guaranteed response time and no dedicated review team.

## Before you start

Read `README.md` for what the app does and `ARCHITECTURE.md` for how the RAG
pipeline is structured. `DECISIONS.md` records the real engineering decisions
made on this project (including things that were tried and rejected, like
LangChain) — worth checking before proposing a significant design change, so
you're not re-litigating something already decided for a documented reason.

## Development setup

```bash
git clone https://github.com/piyushgargog/DocuLens.git
cd ai-document-assistant
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements-dev.txt   # includes requirements.txt + pytest
cp .env.example .env   # then fill in LLM_API_KEY
```

Run the app locally with:

```bash
uvicorn main:app --reload
```

Then open `http://localhost:8000`.

## Project structure

Each pipeline stage is its own small module — keep that separation when
making changes rather than merging responsibilities:

| File | Responsibility |
|---|---|
| `pdf_loader.py` | Page-aware PDF text extraction |
| `chunker.py` | Splitting page text into chunks |
| `embedder.py` | Text -> vector embeddings |
| `vector_store.py` | FAISS similarity search |
| `llm_client.py` | LLM API call + grounding prompt |
| `pipeline.py` | Orchestrates the above (ingest / retrieve / answer / summarize) |
| `main.py` | FastAPI backend (HTTP API, sessions, limits, error shaping) only |
| `static/` | Frontend — plain HTML/CSS/JS, no framework, no build step |
| `evaluate.py` | Reproducible chunking/retrieval comparison |
| `DESIGN.md` | The frontend's design direction — read before changing `static/` |

Keep the pipeline (`pdf_loader.py` through `pipeline.py`) free of any web
framework concerns — `main.py` is the only file that should import
`fastapi`. This is what lets `evaluate.py` and the `tests/` suite call the
pipeline directly, with no server needed.

The project deliberately avoids heavyweight frameworks (see `DECISIONS.md`
for why LangChain/LangGraph aren't used) in favor of small, explicit,
readable functions. Please keep new dependencies to a minimum, and justify
any addition to the pipeline in the PR description.

## Testing your changes

There's a small `pytest` suite (`tests/`) covering PDF extraction, chunking,
embeddings, FAISS retrieval, the LLM client, the full pipeline,
prompt-injection resistance, and the FastAPI HTTP layer (`test_api.py`,
using `TestClient` — no server needs to be running):

```bash
pytest -v
```

Tests that call a real LLM API (`llm_client`, end-to-end `pipeline`, and the
prompt-injection tests) automatically skip if `LLM_API_KEY` isn't set — so
`pytest` still runs meaningfully without credentials, just with less
coverage. A GitHub Actions workflow (`.github/workflows/tests.yml`) runs this
suite on every push/PR to `main`; since no API key secret is configured
there, only the non-LLM tests actually execute in CI — the LLM-dependent
ones skip there too. This is deliberate, not an oversight: it keeps CI
credential-free.

What's **not** automated, and still needs manual verification before opening
a PR:

- Run `uvicorn main:app --reload`, open `http://localhost:8000`, upload
  `sample_docs/sample.pdf`, and confirm the full flow works: an answerable
  question whose yellow page tab highlights the right margin note, a
  follow-up question, a clearly unanswerable question (refused), adding a
  second PDF, a summary, reloading the page (session restored), and
  removing one / all documents. If you touched `static/`, also check a
  375px-wide viewport, dark mode, and keyboard access (Tab reaches the
  upload area). Ask several long questions and check that each answer
  opens at its question (not scrolled to the bottom) and that, on a phone,
  the ask box stays on screen as the conversation grows. The frontend has
  no automated test coverage.
- For a release, bump `APP_VERSION` in `main.py` and the version in
  `static/index.html` (the `?v=` query on `style.css` and `app.js`, so
  browsers that cached the previous release fetch the new files, plus the
  footer label and its release link). `tests/test_api.py` fails if any of
  them disagree.
- CI also runs `pip-audit` on the installed dependencies. If it fails,
  update the affected package (Dependabot usually already has a PR).
- If your change touches retrieval or chunking, run `python
  retrieval_eval.py` and compare Hit@4 / MRR with
  `reports/retrieval_eval.md` (no API key needed). If it touches the
  similarity floor (`RETRIEVAL_SCORE_FLOOR`) or the embedding model, also run
  `python score_floor_eval.py` and check that answerable questions are still
  almost never refused (`reports/score_floor_eval.md`). If it touches the prompt,
  also run `evaluate.py` against a document and question set and check the
  comparison report for regressions:
  ```bash
  python evaluate.py --pdf sample_docs/sample.pdf --questions sample_docs/dev_eval_questions.json
  ```
- Never commit `.env`, an API key, or any other secret. `.env` is gitignored
  — if your diff touches it, something is wrong.

## Documentation

If a change affects user-facing behavior, update `README.md`. If it changes
the pipeline or a design decision, update `ARCHITECTURE.md` and add a dated
entry to `DECISIONS.md` describing what changed and why — this project keeps
an honest, real-time decision log rather than reconstructing rationale after
the fact, and PRs are expected to keep that practice going.

## Submitting a pull request

- Use the PR template.
- Keep PRs focused — one concern per PR is easier to review than a large
  mixed change.
- Describe how you tested the change (see above).

## Reporting bugs or requesting features

Use the issue templates. For anything security-related, do **not** open a
public issue — see `SECURITY.md`.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Please read
it before participating.
