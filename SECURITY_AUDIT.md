# Security audit: DocuLens v4 hardening pass

- **Date:** 2026-10-05
- **Base commit:** `b988c97c07d91ffe2c4331d521d2f7143f93541f` (branch `feat/v4`; the hardening changes are in the commits that follow it)
- **Scope:** the application (`*.py`, `static/`), its tests, the GitHub workflows, both Dockerfiles, dependencies (`requirements*.txt`, `tools/firebase`) and the git history. The live site was checked passively (headers) only; nothing was attacked in production.
- **Result:** `python scripts/release_gate.py security-results.json` -> **PASS** (machine-readable record: [`security-results.json`](security-results.json)).

This is evidence of what was tested and found, not a claim that the application has no vulnerabilities. A clean scanner proves nothing about what it cannot see.

## Status

| | |
|---|---|
| Critical | 0 unresolved |
| High | 0 unresolved (1 found and fixed: F-12) |
| Medium | 9 fixed, 2 accepted with an owner and a review date (F-17, F-19) |
| Low | 6 fixed or false positive, 1 accepted (F-18) |
| Secrets | 0 (history of 102 commits and tracked files) |
| Failing tests | 0 (718 unit/property/fuzz + 10 browser) |
| Dependency advisories | 0 (pip-audit, OSV, npm audit, Trivy) |
| Known auth bypass / cross-user access | 0 |

## Tools, versions and results

| Tool | Status | Version | Notes |
|---|---|---|---|
| pytest | PASS |  | 718 passed (unit, security regression, property-based, API fuzzing) |
| playwright-e2e | PASS |  | 10 passed |
| bandit | PASS | 1.9.4 | 0 issues after fixes (initial: 1 high B324, 3 low) |
| ruff | PASS | 0.16.10 | rules E9,F,B,S |
| mypy | PASS | 2.4.0 | 19 modules, 0 errors (found F-11) |
| semgrep | PASS | 1.x | 81 files; 8 findings: 6 fixed, 2 false positives (F-16); Dockerfile parser unsupported |
| pip-audit | PASS |  | no known vulnerabilities in the installed environment |
| osv-scanner | PASS |  | requirements.txt, requirements-dev.txt, tools/firebase lockfile: clean after F-13/F-14 |
| trivy-fs | PASS | 0.75.0 | 0 HIGH/CRITICAL (vuln, secret, misconfig) |
| gitleaks | PASS | 8.30.1 | history: 102 commits, 0 leaks; working tree: only the untracked, gitignored local .env and a third-party source map |
| actionlint | PASS | 1.7.12 |  |
| zizmor | PASS | 1.30.1 | after F-12 (offline mode) |
| hadolint | PASS | 2.x | Dockerfile, deploy/Dockerfile.update; ignored rules justified in .hadolint.yaml |
| schemathesis | PASS | 4.29.3 | 13/13 operations, 172 cases, 0 failures (sign-in configured) |
| hypothesis | PASS | 6.168.4 | 22 properties; found F-04 and the '..' filename case |
| github-code-scanning | PASS |  | CodeQL default setup; 0 open alerts, 0 Dependabot alerts, 0 secret-scanning alerts (gh api) |
| trivy-image | BLOCKED |  | Docker daemon not running on the audit machine; runs in container-security.yml |
| zap-baseline | BLOCKED |  | no Docker/ZAP locally; runs weekly in dynamic-security.yml |
| sbom | BLOCKED |  | Syft runs in container-security.yml |
| scorecard | BLOCKED |  | runs on GitHub in scorecard.yml |
| live-passive | PASS |  | production headers checked (CSP, HSTS, nosniff, frame, referrer, COOP/CORP); no active testing against production |

`BLOCKED` means it could not run on the audit machine (no Docker daemon) and is run, and gated, by a GitHub workflow instead: it is **not** counted as passed here. The gate accepts `BLOCKED` only for scanners listed in `blocked_with_ci_coverage`.

## Findings

| ID | Severity | Finding | Status | Fix | Regression test |
|---|---|---|---|---|---|
| F-01 | medium | A .docx zip bomb (a few KB inflating to GBs) was parsed without limits | fixed | document_loader.docx_is_safe checks entry count, per-part and total size and ratio from the zip directory before any parsing | `tests/test_security_hardening.py::TestUploads::test_a_docx_zip_bomb_is_refused_before_it_is_decompressed` |
| F-02 | medium | JSON request bodies were buffered in full before the size cap was applied (chunked transfer, no Content-Length) | fixed | main._json_body streams the body and stops at MAX_JSON_BYTES | `tests/test_security_hardening.py::TestRequests::test_oversized_json_is_refused_without_buffering_it` |
| F-03 | medium | Parallel uploads could exceed the per-user / guest document cap (check-then-act race) | fixed | the cap check, the save and the registration run under one per-session lock; mutation-tested (7 stored with the lock removed, cap 5) | `tests/test_security_hardening.py::TestConcurrency::test_parallel_uploads_by_one_user_never_exceed_the_cap_or_corrupt_the_index` |
| F-04 | medium | Streaming parsers crashed on well-formed JSON events that are not objects (null, array, number) | fixed | llm_adapters._json_object skips non-object events | `tests/test_properties.py::test_valid_event_json_of_any_shape_never_crashes_the_stream` |
| F-05 | medium | A corrupt value in Redis raised through the state store and tripped the 30 s circuit breaker for everything | fixed | RedisStore drops undecodable values/fields and reports them as absent; the breaker is for outages only | `tests/test_security_hardening.py::TestStateStore::test_corrupt_values_are_dropped_without_tripping_the_circuit_breaker` |
| F-06 | low | Application shutdown raised if Redis was down | fixed | ResilientStore.close swallows and logs the error | `tests/test_security_hardening.py::TestStateStore::test_during_an_outage_a_signed_in_users_documents_are_hidden_not_exposed` |
| F-07 | low | Every token with an unknown key id made the server fetch Google's certificates (outbound amplification) | fixed | refetch at most once per 60 s | `tests/test_auth.py::test_unknown_key_id_is_rejected` |
| F-08 | low | Filenames could carry bidirectional-override / zero-width characters, backslash paths or a bare '..' | fixed | main._clean_filename strips them | `tests/test_properties.py::test_clean_filename_is_always_a_short_inert_basename` |
| F-09 | medium | Prompt-structure forgery: fence variants (case, spacing, full-width), fake '[Page N]' labels and bracket-closing filenames | fixed | llm_client._neutralize defuses any run of 3+ angle brackets and [Page N] lookalikes in document text, history and filenames | `tests/test_security_hardening.py::TestPromptInjectionStructure` |
| F-10 | medium | A saved HNSW index was deserialised by FAISS with only a digest stored next to it | fixed | index blobs are stored only with DOCULENS_STATE_KEY and verified with HMAC-SHA256; otherwise rebuilt from the embeddings | `tests/test_vector_index.py::test_a_saved_index_signed_with_another_key_is_never_deserialised` |
| F-11 | medium | A document removed on another instance between the membership check and the load raised IndexError (500) | fixed | main._one_document returns 404 | `tests/test_security_hardening.py::TestConcurrency::test_deleting_while_asking_never_crashes` |
| F-12 | high | GitHub Actions referenced by mutable tags; checkout persisted credentials (zizmor: 4 high, 2 low) | fixed | every action pinned to a full commit SHA, persist-credentials: false, minimal permissions | `zizmor + actionlint in .github/workflows/security.yml` |
| F-13 | medium | Dependency floors allowed versions with published advisories (pyjwt, anyio, idna, torch, transformers, pygments) | fixed | requirements minimums raised; pip-audit and OSV re-run clean | `pip-audit step in tests.yml` |
| F-14 | low | firebase meta-package pulled firestore/grpc (4 high npm advisories) although only auth is bundled | fixed | tools/firebase now depends on @firebase/app and @firebase/auth only; npm audit: 0 | `npm audit (tools/firebase)` |
| F-15 | low | SHA-1 used for an in-memory cache key (bandit B324 high); silent except-pass in pdf_loader (B110/B112) | fixed | BLAKE2b; the exceptions are logged | `bandit clean` |
| F-16 | low | Semgrep ssrf-injection-requests on llm_client._post (variable named 'request') | false_positive | the URL is built from operator configuration only; renamed and a host-equality check added as defence in depth | `tests/test_security_hardening.py::TestOutboundRequests` |
| F-17 | medium | CSP must allow https://apis.google.com for Firebase's popup sign-in | accepted | applies only when Firebase is configured; every DOM sink is textContent (tested), no inline script/eval, object-src and base-uri none | `tests/test_security_hardening.py::TestFrontendSinks` |
| F-18 | low | A stolen Firebase ID token can be exchanged for a login for up to an hour (no jti to revoke) | accepted | inherent to bearer tokens; /api/login is rate limited, our own session is separately revocable (logout) | `tests/test_auth.py` |
| F-19 | medium | A PDF 'flate bomb' can exhaust memory inside PyMuPDF text extraction, which cannot be bounded in-process | accepted | container memory limit and restart policy (deploy command, container-security.yml); page and time limits; recommended: run behind --memory | `tests/test_security_hardening.py::TestUploads::test_a_pdf_with_an_absurd_page_count_is_bounded` |

Reproduction for each is its regression test: it fails against the pre-fix code (F-03 was mutation-checked: with the lock removed 7 documents were stored against a cap of 5).

## Accepted risks (owner: Piyush Garg, review by 2026-12-05)

- **F-17 CSP allows `https://apis.google.com` when Firebase sign-in is configured.** That host has served JSONP endpoints that can defeat a strict CSP *if* an XSS bug exists. Mitigations: no `innerHTML`/inline script/eval anywhere (enforced by tests), all untrusted text goes through `textContent`, `object-src 'none'`, `base-uri 'none'`, and the widening applies only when Firebase is configured.
- **F-18 ID-token replay for up to one hour.** Firebase ID tokens have no revocable id; a stolen one can start a session until it expires. `/api/login` is rate-limited and our session is revocable by logout.
- **F-19 PDF decompression bombs inside PyMuPDF.** Page count, extraction time, OCR pixels/pages/time and upload size are bounded, but memory used *inside* one extraction call is not. Run the container with a memory limit (the CI boot test and the deploy command use one) so such a file kills only that request's worker, not the host.

## Not covered / limits

- No end-to-end Firebase popup sign-in against a real project (needs the owner's project); the verification path is tested with real RS256 tokens and a fake certificate endpoint.
- OCR is tested with a scripted engine; real scans were not available here.
- Prompt injection: the structure defences are tested offline; the model-behaviour tests (`test_prompt_injection.py`) need an API key and were not run in this pass. Prompt injection is mitigated, not solved.
- ZAP, Trivy image, SBOM and Scorecard run in GitHub workflows and have not produced results yet.
- Upstash and Firebase are third parties: their availability and security are outside this audit.

## Release decision

The release gate passes with the blocked scanners covered by CI. Release also requires those workflows to be green on the pull request.
