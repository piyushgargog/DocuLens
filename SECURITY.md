# Security Policy

## Supported versions

Only the latest release, the one running at https://doculens.duckdns.org, is
supported. Fixes land on `main` and ship as a new release.

## What the project already does

The summary is below; the full list of controls is in `README.md` (Security)
and `ARCHITECTURE.md` (Security controls).

- **Prompt injection:** passages are fenced and declared untrusted, and fence
  tokens in uploaded text, filenames and earlier turns are neutralised. Tests
  cover the main and fallback models.
- **Secrets:** keys are read from the environment only and are never logged,
  rendered or committed. `/api/status` shows provider names and availability,
  not keys.
- **Sessions:** `__Host-` cookies (HttpOnly, SameSite=Lax, Secure): documents
  under a sliding 2-hour session, optional Firebase sign-in (ID token verified
  server-side: RS256 signature, audience, issuer, expiry, verified e-mail) under
  a 7-day login. Documents are tied to the browser session, not the account.
  Cross-site API requests are refused.
- **Data:** the original upload is never stored. A guest's text lives in memory until
  removal, 2 hours idle, or restart; a signed-in user's text and embeddings are
  stored in Redis under a namespace derived from the verified uid, for 30 days or
  until removed. Only the question and retrieved passages
  are sent to the AI provider, and the upload screen says so. Model reasoning
  is displayed but never stored.
- **Limits:** per-IP rate limits (`X-Real-IP` is trusted only from
  `TRUSTED_PROXIES`), 25MB uploads, a ~360-page cap checked before embedding,
  30-page OCR, 5 documents per session, 50 sessions, `MAX_TOTAL_CHUNKS`, 16KB
  JSON bodies, and caps on concurrent work.
- **Web:** strict CSP and security headers, HSTS, API docs disabled, all
  dynamic text via `textContent`, no third-party requests.
- **Supply chain:** `pip-audit` and CodeQL in CI, plus Dependabot.
- **Deployment:** the container runs as a non-root user (uid 10001) and binds
  to `127.0.0.1`. Nginx terminates TLS 1.2/1.3, sets `X-Real-IP`, and answers
  HTTPS only for the site's hostname.

## Reporting a vulnerability

Do **not** open a public issue. Contact the maintainer privately via GitHub,
[@piyushgargog](https://github.com/piyushgargog). Include the impact, steps to
reproduce (a crafted file or prompt, never real personal data), and the
affected part (ingestion, retrieval, LLM call, UI).

## Scope

In scope: this app's code, including prompt construction, upload handling,
secret handling, the backend and the frontend. Out of scope: third-party LLM
providers' models and infrastructure; report those to the provider.

## Response

This is a spare-time project with no SLA. Security reports are handled before
features and general bugs.
