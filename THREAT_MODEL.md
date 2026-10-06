# Threat model

## Assets
User documents (extracted text and embeddings), questions and answers, login sessions, provider and Redis credentials, the availability of a 2 vCPU / 2 GB host, and the free-tier LLM quotas.

## Trust boundaries and data flows

```
browser --HTTPS--> Nginx (TLS, sets X-Real-IP) --127.0.0.1--> FastAPI app
   |                                                         |-- Redis (Upstash, TLS): logins, rate limits, saved documents
   |-- Firebase SDK --> Google (sign-in popup)               |-- LLM providers (HTTPS): question + 4 passages, or summary batches
   `-- uploads (PDF/DOCX/TXT/MD)                             `-- Google certificates (HTTPS): verify Firebase ID tokens
CI: GitHub Actions (pinned actions, read-only token) -> tag -> manual deploy over SSH to EC2
```

User-controlled inputs: upload bytes and filename, JSON bodies (`question`, `doc_ids`, `id`, `id_token`), cookies, headers (only `X-Real-IP` from a trusted proxy is used), multipart field names. Document text, filenames and conversation history are **untrusted** everywhere they reach a prompt.

Untrusted data reaches: the filesystem (never; uploads are processed in memory, `ARTIFACT_DIR` filenames are hashes), subprocesses (Tesseract via pytesseract, language setting validated), HTML (only `textContent`), Redis (keys built from fixed prefixes, a SHA-256 of the verified uid and server-generated ids), URLs (none: outbound hosts come from operator configuration), prompts (fenced and neutralised), logs (no content, tokens or keys).

## Authentication and authorization
Identity is a Firebase ID token verified server-side (RS256 only, audience, issuer, expiry, verified e-mail) and exchanged once for our own HttpOnly `__Host-` cookie backed by a Redis entry. Every document, history and quota key is namespaced by `sha256(uid)` derived from that verified identity; a document id from the browser is only ever looked up inside the caller's own list. Guests are identified by IP for quotas and by an unguessable session cookie for their single in-memory document.

## Threats and controls (STRIDE-style summary)
| Threat | Control | Tested in |
|---|---|---|
| Forged / replayed / wrong-audience token | signature + claims checks, cert cache with refetch throttle, rate limit | test_auth, test_security_hardening |
| Cross-user access (IDOR/BOLA) | owner namespace from verified uid, membership checks, no client-supplied identity | TestAuthorization |
| XSS / HTML injection | `textContent` only, strict CSP, no inline script | TestFrontendSinks |
| CSRF | SameSite=Lax + Origin / Sec-Fetch-Site refusal on every state-changing API call | TestRequests |
| Prompt injection (direct, via documents, filenames, history) | fenced untrusted passages, neutralised delimiters and fake labels, reminder after the question, history labelled untrusted and never used as evidence, rewrite validated | TestPromptInjectionStructure, test_prompt_injection |
| Resource exhaustion | upload/JSON/page/chunk/OCR/summary budgets, bounded concurrency, per-IP and per-user rate limits, zip-bomb guard | TestUploads, test_api_fuzz |
| Redis outage / corruption | circuit breaker with in-process fallback, corrupt values dropped, fail closed for identity | TestStateStore |
| SSRF | no user-controlled URL; provider host equality check; operator URLs validated | TestOutboundRequests |
| Supply chain | pinned Actions (SHA), Dependabot with cooldown, pip-audit/OSV/Trivy, SBOM, minimum versions past advisories | security.yml, container-security.yml |
| Stolen deployment access | SSH restricted by security group, non-root container, no secrets in the image | deployment notes in SECURITY.md |

## Out of scope / residual
See "Accepted risks" in `SECURITY_AUDIT.md`.
