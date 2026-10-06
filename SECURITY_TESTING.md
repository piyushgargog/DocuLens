# Security testing

## What runs where

| Layer | Tool | Where |
|---|---|---|
| Unit, regression, integration | pytest (718 tests) | every push (`tests.yml`) |
| Security regression corpus | `tests/test_security_hardening.py` | every push |
| Property-based | Hypothesis, `tests/test_properties.py` (22 properties) | every push |
| API fuzzing | `tests/test_api_fuzz.py` (every route from the OpenAPI document) + Schemathesis CLI | every push / `security.yml` |
| Browser | Playwright, `tests/e2e` (10 tests) | every push |
| Static analysis | ruff (E9,F,B,S), mypy, bandit, Semgrep, CodeQL (default setup) | `tests.yml`, `security.yml`, GitHub |
| Secrets | gitleaks (full history), GitHub secret scanning + push protection | `security.yml`, GitHub |
| Dependencies | pip-audit, Trivy fs, `dependency-review.yml`, Dependabot (5-day cooldown) | CI |
| Workflows | actionlint, zizmor | `security.yml` |
| Container | hadolint, Trivy image, Syft SBOM, hardened-boot test | `container-security.yml` |
| Dynamic | OWASP ZAP baseline (passive, local instance) | `dynamic-security.yml` (weekly) |
| Posture | OpenSSF Scorecard | `scorecard.yml` |
| Gate | `scripts/release_gate.py` over `security-results.json` | before a release |

## Adding a regression test for a security bug
1. Reproduce it as a failing test in `tests/test_security_hardening.py` (or a Hypothesis property).
2. Fix the root cause. 3. Record the finding in `security-results.json` with `regression_test` set. 4. Run `python scripts/release_gate.py`.

## Rules
Never silence a scanner without recording why; never mark a test skipped to get green; a scanner that did not run is `BLOCKED`, not `PASS`.

## Running locally
```
pip install -r requirements-dev.txt
pytest                                   # everything except the browser tests
DOCULENS_E2E=1 pytest tests/e2e          # browser tests
ruff check . && mypy main.py ...         # see tests.yml for the module list
bandit -r . -x ./tests,./venv,./tools,./static
python scripts/release_gate.py
```
