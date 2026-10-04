# AI Usage — DocuLens

How an AI coding assistant was used to build this project. The detailed,
dated record of each change is in `DECISIONS.md`.

## Division of work

- **The user** set the requirements, the stack and the workflow (plan before
  code, phased build), chose among proposed options, supplied credentials and
  reference material (a security checklist, design tokens, real test
  documents), and reviewed every release.
- **The assistant** drafted the planning documents, most of the code, the tests
  and the docs. It proposed architecture choices with reasons for the user to
  accept or reject, ran and verified everything, and deployed to EC2 over a
  scoped SSH key.
- Keys supplied by the user went only into the gitignored `.env`. They were
  never printed, hardcoded or committed.

## How results were kept honest

- Nothing was reported as working until it had been run. Evaluation numbers in
  the README and reports are actual script output.
- Suspicious results were checked before they were reported. For example, a
  "wrong" refusal turned out to be correct, and an earlier "misread BLEU"
  claim was found to be wrong and was corrected.
- When the main model's quota ran out, results were labelled with the model
  that actually produced them.
- Outside code reviews and proposed pull requests were treated as claims to
  verify, not as instructions. Parts that held up were kept; others were
  rejected with reasons. One proposal removed a feature while claiming no
  breaking changes; another had a prefetch that recursed forever.
- The assistant's own mistakes are recorded rather than hidden: a deleted
  image during an early deploy, a hardcoded sample on the landing page, a
  "Go deeper" prompt that padded answers with outside knowledge, and a
  similarity floor (v3.6.0) that refused real resume questions and is now off
  by default.

## Major milestones

| Area | What the assistant did |
|---|---|
| Pipeline and evaluation | Built the RAG pipeline and `evaluate.py`, then `retrieval_eval.py` (Hit@k/MRR against BM25 and BGE baselines), and moved to hybrid retrieval only after measuring it |
| UI | Replaced Streamlit with FastAPI and a static frontend for mobile, then redesigned it on the user's direction, ending in "Calm Light" from the user's reference tokens; verified in a headless browser each time |
| Deployment | Compared hosts by testing real limits, deployed to EC2 with Docker, Nginx and HTTPS, and found the CPU-only PyTorch fix from the build log |
| Security | Audited against the user's checklist, the git history and the live site; added CSP and headers, rate limits, CSRF and cookie hardening, prompt-injection fences and `pip-audit` |
| Reliability | Added provider failover with cooldowns, streaming, OCR, observability, and v3.6 hardening (trusted proxies, locks, memory budget) |
| Accessibility | Keyboard access, focus visibility, screen-reader announcements and reduced-motion handling, adjusted to the user's feedback |

## What the author must be able to explain

- **Chunking:** a character window per page with overlap, and why chunk size
  mattered most in the A/B test.
- **Retrieval:** normalised MiniLM vectors (inner product = cosine), FAISS
  exact search, BM25 + RRF fusion, and overview routing.
- **Grounding:** fenced untrusted passages, the exact refusal sentence, and
  why the similarity floor is off.
- **Multi-document and follow-ups:** one index per document; the previous
  question joins retrieval but is never used as evidence.
- **Retention and safety:** nothing written to disk, the in-memory session
  lifecycle, and `textContent`-only rendering.
- **Why no LangChain:** see `DECISIONS.md`.
