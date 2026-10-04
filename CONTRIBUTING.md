# Contributing to DocuLens

This is a personal project maintained by Piyush Garg in spare time. Issues and
PRs are welcome, but there is no guaranteed response time.

## Before you start

Read `README.md` and `ARCHITECTURE.md`. Check `DECISIONS.md` before proposing a
design change; many alternatives (LangChain, sentence-aware chunking, a
similarity floor) were already tried and the outcome is recorded there.

## Setup

```bash
git clone https://github.com/piyushgargog/DocuLens.git
cd DocuLens
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env              # add at least one provider key
uvicorn main:app --reload
```

## Ground rules

- Keep one responsibility per module. `main.py` is the only file that imports
  FastAPI, so the pipeline, `evaluate.py` and the tests run without a server.
- Keep dependencies minimal and explain any new one in the PR.
- Read `DESIGN.md` before changing `static/`.
- Never commit `.env` or any secret.

## Testing

- `pytest` runs the suite. Tests that call a real LLM skip without a key, so
  CI stays credential-free.
- CI also runs a headless-browser end-to-end test (upload, ask, citation chips,
  refusal, keyboard access), `pip-audit` and CodeQL.
- If you change `static/`, also check a 375px viewport, dark mode, keyboard
  navigation and screen-reader announcements.
- If you change retrieval or chunking, run `python retrieval_eval.py` and
  compare against `reports/retrieval_eval.md`.
- If you change a retrieval threshold or the embedding model, also run
  `python score_floor_eval.py` and try a short real document such as a resume.
- If you change a prompt, run `evaluate.py` on a document and question set.
- For a release, bump `APP_VERSION` in `main.py` and the version strings in
  `static/index.html` (asset `?v=`, footer, release link). A test checks they
  match.

## Documentation

Update `README.md` for user-facing changes and `ARCHITECTURE.md` for pipeline
changes. Add a short dated entry to `DECISIONS.md` for any design decision.
Keep the docs concise.

## Pull requests and issues

Use the templates, keep each PR focused, and say how you tested it. Report
security issues privately (see `SECURITY.md`). This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
