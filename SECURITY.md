# Security Policy

## Supported versions

Releases are tagged on `main` (`vX.Y.Z`, see the GitHub Releases page),
but there are no maintained release branches. Only the latest release —
which is what runs at https://doculens.duckdns.org — is
supported. Security fixes land on `main` and ship as a new patch release.

## What this project already does

For context before reporting an issue, see the "Security notes" section of
`README.md`. In short:

- Retrieved document passages are fenced in the LLM prompt and declared
  untrusted data. Delimiter tokens inside passage text and uploaded filenames
  are neutralized before prompt construction, and summary/suggestion prompts
  carry the same untrusted-content reminder. Indirect prompt-injection tests
  cover both normal and fallback models.
- LLM provider keys are read only from the environment (`GEMINI_API_KEY`,
  `GROQ_API_KEY` or `LLM_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `HF_TOKEN`); the
  public `/api/status` shows provider names, models and availability only.
  Keys are never
  logged, rendered, or committed. `.env` is gitignored; only `.env.example`
  (placeholders) is tracked.
- The session cookie is `httponly`, `samesite=lax`, and `Secure` over HTTPS;
  active API requests refresh its expiry. It is an anonymous bearer session,
  not authentication or an account boundary.
- Error responses never include exception text or stack traces; details are
  logged on the server only.
- Uploaded PDFs are never written to disk; their extracted text lives only
  in memory and is deleted on removal or after 2 hours of inactivity (a
  background task enforces this every 5 minutes).
- Resources are bounded: 25MB uploads (never read past the limit), 1000-
  character questions, 5 documents per session, 50 sessions, a configurable
  75,000-chunk aggregate index budget, and 2-hour expiry.
- Document and model text is inserted into the page with `textContent` /
  text nodes only, never as HTML or markdown.
- Strict Content-Security-Policy and other security headers on every
  response; HSTS over HTTPS; API docs endpoints disabled; no third-party
  requests from the page (fonts are self-hosted).
- Per-client rate limits on LLM-backed endpoints and uploads, and caps on
  concurrent LLM calls and ingestions. The client IP is taken from `X-Real-IP`
  only when the immediate peer matches a configured `TRUSTED_PROXIES` CIDR;
  loopback is the safe default and Docker bridge networks must be explicit.
- Cross-site API requests are refused (Origin / Sec-Fetch-Site), the
  session cookie is `__Host-` prefixed over HTTPS, and JSON bodies are
  capped at 16KB.
- CI runs `pip-audit`; a known vulnerability in any installed dependency
  fails the build and blocks merging.
- Uploads are checked for a real PDF signature and sanitised filenames.
  Oversized documents are rejected before embedding, a best-effort aggregate
  chunk budget limits total in-memory indexes, and OCR of scanned PDFs is bounded
  (at most 30 pages) so one upload cannot turn into unbounded CPU work.
  Retrieval also abstains when all non-overview hits fall below the calibrated
  score floor.
- The upload screen discloses that questions and relevant passages are sent
  to the LLM provider.
- Dependencies are watched by Dependabot, and CodeQL scans every push.
- The container runs as an unprivileged user (uid 10001), not root, so a
  hypothetical code-execution bug in a dependency is not already root inside
  the container (v2.2.2).
- In deployment the app binds only to `127.0.0.1`; Nginx terminates TLS
  (1.2/1.3 only), overwrites `X-Real-IP` with the real peer so the rate limit
  can't be spoofed, and answers HTTPS only for the site's own hostname. Because
  Docker's port mapping makes Nginx appear as the bridge gateway, the app's
  `TRUSTED_PROXIES` lists exactly that address (plus loopback); a request from
  any other address has its `X-Real-IP` ignored.
- Model "thinking" streamed by some providers is display-only: it is never
  stored in the conversation history, never sent back to the model, and is
  discarded if the route that produced it fails.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security vulnerability.

GitHub's private vulnerability reporting is not currently enabled on this
repository, so please report privately by contacting the maintainer
directly via GitHub —
[@piyushgargog](https://github.com/piyushgargog) — instead of filing a
public issue.

Please include:
- A description of the issue and its potential impact.
- Steps to reproduce, or a minimal example (a crafted PDF or prompt, for
  example — not real personal data).
- Which part of the app is affected (ingestion, retrieval, the LLM call, the
  UI, etc.).

## Scope

In scope: this application's own code — prompt construction and grounding,
handling of uploaded files, secret handling, the FastAPI backend
(`main.py`), and the frontend (`static/`).

Out of scope: the underlying third-party LLM provider's model behavior or
infrastructure (e.g. Groq, OpenAI, or any other OpenAI-compatible endpoint
you configure). Report those directly to the provider.

## Response expectations

This is a small project maintained in spare time — there's no guaranteed
response time or SLA, but security reports will be prioritized over feature
requests and general bugs.
