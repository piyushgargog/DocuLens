# Project Specification — DocuLens

This document defines the requirements this project is built to satisfy: a
small, self-contained RAG (Retrieval-Augmented Generation) tool for asking
grounded questions about uploaded PDFs. It started as a single-document
tool; the optional enhancements listed originally (multiple documents,
conversation history, document summary) have since been built and are now
part of the requirements below. _Current as of v3.6.0._

## Problem

A user has a document and wants to ask natural-language questions about its
contents without reading the whole thing. Plain keyword search doesn't
handle paraphrased questions, and a raw LLM call without the document will
hallucinate answers not actually present in the source.

## Objective

Build a document question-answering tool that:
- Ingests PDFs supplied by the user at runtime (no document-specific setup).
- Answers questions using only information retrieved from those documents.
- Shows exactly which document, page and passage each answer came from.
- Refuses to answer when the documents don't contain the answer.

## Scope

### Core requirements
- PDF upload with page-aware text extraction and chunking.
- Embedding-based retrieval of relevant passages per question.
- Grounded answer generation (LLM restricted to retrieved context).
- Source page and passage shown with every answer.
- A test set of ≥5 questions, including ≥1 unanswerable from the document.
- Two chunking/retrieval configurations compared, with results recorded in
  `DECISIONS.md`.

### Enhancements (built)
- Up to 5 documents per session, searched together, with every source
  labeled by document and page.
- Follow-up questions: recent turns are used to resolve references, never
  as a source of facts.
- A short, page-cited summary of any loaded document.

### Explicitly not building
- User accounts, persistence across server restarts, or a database — state
  is an in-memory session per browser.
- Fine-tuning any model.
- A frontend framework, bundler or build step — the frontend is plain
  HTML/CSS/JS.

## Functional Requirements

| ID | Requirement | Rationale |
|----|-------------|--------|
| FR1 | Load a PDF supplied by the user. | Core input to the whole pipeline. |
| FR2 | Extract text per page and split into searchable chunks, retaining page numbers. | Page-level provenance is required for citations (FR6). |
| FR3 | Embed chunks and index them for similarity search. | Semantic, not just keyword, retrieval. |
| FR4 | Given a question, retrieve the top-k most relevant chunks. | Standard RAG retrieval step. |
| FR5 | Generate an answer with an LLM constrained to the retrieved chunks. | Grounding — prevents hallucinated answers. |
| FR6 | Show the source document, page and passage text for each answer. | Lets the user verify the answer. |
| FR7 | If no retrieved chunk supports an answer, say so rather than guess. | A conservative refusal beats a confident wrong answer. |
| FR8 | Compare at least two chunking/retrieval configurations without code changes. | Parameters materially affect answer quality (see `DECISIONS.md`); `evaluate.py` reruns the comparison on any PDF. |
| FR9 | Load up to 5 documents at once; search across all of them; remove one or all. | Questions often span several documents. |
| FR10 | Resolve follow-up questions ("how many moons does it have?") from recent turns, without using earlier answers as evidence. | Natural conversation without weakening grounding. |
| FR11 | Produce a short summary of a chosen document, citing pages. | Orientation before asking questions. |
| FR12 | Treat document text as untrusted: instructions inside a PDF must not change the assistant's behavior. | Uploaded files are attacker-controllable input to the prompt. |
| FR13 | Answer whole-document questions ("what is this about?") from an overview of the document, not similarity search. | No single passage resembles such a question. |
| FR14 | Explain rather than quote: adapt the level on request, combine passages, and answer partially with what's missing instead of refusing outright. | The LLM's value over retrieval alone. |
| FR15 | Keep answering when the main model's quota runs out (configurable fallback model). | Free-tier daily limits would otherwise take the live app down. |
| FR16 | Stream answers as they are written, and let the user stop one. | Perceived speed; control over long answers. |
| FR17 | Let the user choose which loaded documents a question searches. | Precision with several documents loaded. |
| FR18 | Copy an answer with its sources; export the conversation as Markdown. | Answers are meant to be reused and cited. |
| FR19 | Use the retrieval method that measured best (hybrid BM25 + embeddings). | Retrieval quality is measured, not assumed. |

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | Code small, clean and readable — clarity over premature production hardening. |
| NFR2 | LLM provider, model and API key configurable via environment variables, never hardcoded. |
| NFR3 | Runs locally via `uvicorn main:app` with the setup documented in `README.md`, and deploys via the included `Dockerfile` on any container host. |
| NFR4 | Degrades gracefully (clear message, correct HTTP status, never a crash or a frozen screen) when the API key is missing, the PDF is invalid, the LLM fails, or a proxy returns an error page. |
| NFR5 | No fabricated test results — everything in `DECISIONS.md`/README reflects actual runs. |
| NFR6 | Usable at a 375px mobile width with no horizontal overflow; light and dark themes; keyboard-operable (including upload); respects reduced motion. |
| NFR7 | Bounded resources: 25MB uploads, 1000-character questions, 5 documents per session, 50 sessions, 2-hour idle expiry enforced by a background sweep. |
| NFR8 | No document-specific content hardcoded in the application source. |
| NFR9 | Uploaded documents are never persisted: no disk writes, and their extracted text is deleted on removal or expiry. |
| NFR10 | Abuse-resistant: per-client rate limits, real-PDF check, document size cap before embedding, strict CSP and security headers, no public API docs, errors carry a reference instead of internals. |
| NFR11 | Users are told what leaves the server: questions and retrieved passages go to the LLM provider; the page itself contacts no third party. |

## Acceptance Criteria

- **Document processing** — text is extracted per page and chunks keep page
  (and document) metadata; verified with valid, invalid and empty PDFs.
- **Retrieval quality** — measured, not eyeballed: Hit@1/Hit@4/MRR@10 on a
  39-question labelled set against a BM25 baseline and a second embedding
  model, across three chunk settings (`retrieval_eval.py`,
  `reports/retrieval_eval.md`).
- **Grounding** — answers use only retrieved content; the unanswerable
  question produces the explicit refusal, not a hallucination.
- **Source handling** — every answer shows its supporting passages with
  document, page and similarity score, and page references in the answer
  link to the matching passage.
- **Understanding of the AI pipeline** — extraction → chunking → embedding →
  retrieval → generation are distinct, inspectable functions, explained in
  `ARCHITECTURE.md`.

## Constraints

- Python, PyMuPDF, sentence-transformers, FAISS, a configurable
  OpenAI-compatible LLM API, and a FastAPI backend serving a static
  HTML/CSS/JS frontend.
- Must actually run and be tested before being called done — 146 automated
  tests plus real-browser checks of the deployed app (see `README.md`).
