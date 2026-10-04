# Project Specification — DocuLens

_Current as of v3.7.0._

## Problem

People want to ask questions about a document without reading all of it.
Keyword search misses paraphrases, and an LLM without the document makes up
answers.

## Objective

A document Q&A tool that:
- accepts documents at runtime, with no document-specific setup;
- answers only from those documents;
- shows the document, page and passage behind each answer;
- refuses when the documents don't contain the answer.

## Functional requirements

| ID | Requirement |
|----|-------------|
| FR1 | Load user-supplied documents (PDF, .docx, .txt, .md; OCR for scans). |
| FR2 | Extract text per page and chunk it, keeping page numbers. |
| FR3 | Embed and index chunks for retrieval. |
| FR4 | Retrieve the top-k relevant chunks per question, using hybrid BM25 + embeddings because it measured best. |
| FR5 | Generate answers constrained to the retrieved chunks. |
| FR6 | Show source document, page and passage for every answer. |
| FR7 | Refuse rather than guess when nothing relevant is found. |
| FR8 | Compare chunking/retrieval configurations without code changes (`evaluate.py`). |
| FR9 | Support up to 5 documents; search all or a chosen subset; remove one or all. |
| FR10 | Resolve follow-ups from recent turns without using them as evidence. |
| FR11 | Summarize a document with page citations. |
| FR12 | Treat document text as untrusted: embedded instructions must not be obeyed. |
| FR13 | Answer whole-document questions from an overview sample. |
| FR14 | Explain rather than quote; give partial answers and name what's missing. |
| FR15 | Keep answering when one provider fails (failover chain). |
| FR16 | Stream answers and allow stopping. |
| FR17 | Copy answers with sources; export the conversation as Markdown. |

## Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR1 | Small, readable code; no heavyweight frameworks. |
| NFR2 | Providers, models and keys configured via environment only. |
| NFR3 | Runs with `uvicorn main:app`; deploys via the `Dockerfile`. |
| NFR4 | Clear errors with correct HTTP status codes; the app never crashes and the UI never freezes. |
| NFR5 | No fabricated results; `DECISIONS.md` reflects actual runs. |
| NFR6 | Works at 375px, has light and dark themes, is keyboard- and screen-reader-usable, and respects reduced motion. |
| NFR7 | Bounded resources (upload size, question length, documents, sessions, total chunks, idle expiry). |
| NFR8 | No document-specific content in the source. |
| NFR9 | Uploads are never persisted. |
| NFR10 | Abuse-resistant: rate limits, upload checks, strict CSP and headers, generic errors. |
| NFR11 | Users are told what is sent to the AI provider. |

## Out of scope

User accounts, persistence across restarts, a database, fine-tuning, and a
frontend framework or build step.

## Acceptance

- Retrieval is measured (Hit@k, MRR) against baselines (`reports/retrieval_eval.md`).
- Unanswerable questions produce the exact refusal.
- Every answer shows its passages, and page chips link to them.
- Pipeline stages are separate, testable functions (`ARCHITECTURE.md`).
- Automated tests and a browser end-to-end test pass in CI.
