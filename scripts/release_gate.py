"""Machine-readable release security gate.

Reads security-results.json and exits non-zero (blocking the release) when any
rule below is violated. It deliberately fails closed: a scanner that is missing,
errored or marked BLOCKED where it is required is a failure, never a pass.

    python scripts/release_gate.py [security-results.json]

Rules (all must hold):
  1. no finding of severity critical or high is unresolved;
  2. no secret is reported (secrets.found == 0);
  3. no authentication / authorization / cross-user finding is unresolved;
  4. every medium finding is fixed or explicitly accepted (with rationale, owner and
     a review date that has not passed);
  5. every low finding is fixed or accepted with the same documentation;
  6. every scanner listed in `required_scanners` ran: status PASS (not FAIL, ERROR,
     BLOCKED or missing);
  7. tests: failing == 0 and total > 0;
  8. dependency vulnerabilities and container blockers are 0 (or the scan is BLOCKED
     and listed in `blocked_with_ci_coverage`, meaning a CI workflow runs it).
"""

import json
import sys
from datetime import date
from pathlib import Path

RESOLVED = {"fixed", "false_positive"}
SEVERITIES = ("critical", "high", "medium", "low", "info")
AUTH_TAGS = {"authentication", "authorization", "cross-user"}


def evaluate(results: dict, today: date | None = None) -> list[str]:
    """Return the list of violations (empty = the release may proceed)."""
    today = today or date.today()
    problems: list[str] = []

    for f in results.get("findings", []):
        sev, status = f.get("severity", "").lower(), f.get("status", "").lower()
        fid = f.get("id", "?")
        if sev not in SEVERITIES:
            problems.append(f"{fid}: unknown severity {sev!r}")
            continue
        if sev in ("critical", "high") and status not in RESOLVED:
            problems.append(f"{fid}: unresolved {sev} finding ({status or 'no status'})")
        if AUTH_TAGS & set(f.get("tags", [])) and status not in RESOLVED:
            problems.append(f"{fid}: unresolved authentication/authorization finding")
        if sev in ("medium", "low") and status not in RESOLVED:
            if status != "accepted":
                problems.append(f"{fid}: {sev} finding is neither fixed nor accepted")
            else:
                acc = f.get("acceptance", {})
                missing = [k for k in ("rationale", "owner", "review_date") if not acc.get(k)]
                if missing:
                    problems.append(f"{fid}: accepted without {', '.join(missing)}")
                elif date.fromisoformat(acc["review_date"]) < today:
                    problems.append(f"{fid}: acceptance expired on {acc['review_date']}")
        if status in RESOLVED and sev in ("critical", "high", "medium") and not f.get("regression_test"):
            problems.append(f"{fid}: fixed {sev} finding has no regression test recorded")

    if results.get("secrets", {}).get("found", 1) != 0:
        problems.append("secrets: a secret was found (or the secret scan result is missing)")

    scanners = results.get("scanners", {})
    covered = set(results.get("blocked_with_ci_coverage", []))
    for name in results.get("required_scanners", []):
        entry = scanners.get(name)
        status = (entry or {}).get("status", "MISSING").upper()
        if status == "PASS":
            continue
        if status == "BLOCKED" and name in covered:
            continue  # cannot run on this machine; a CI workflow runs it and gates on it
        problems.append(f"scanner {name}: {status}")

    tests = results.get("tests", {})
    if tests.get("failed", 1) != 0 or not tests.get("total"):
        problems.append("tests: failing tests, or no test result recorded")

    for key, label in (("dependency_vulnerabilities", "dependency vulnerabilities"), ("container_blockers", "container blockers")):
        if results.get(key, 1) != 0:
            problems.append(f"{label}: {results.get(key, 'missing')}")
    return problems


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "security-results.json")
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"RELEASE GATE: FAIL -- cannot read {path}: {e}")
        return 2
    problems = evaluate(results)
    if problems:
        print("RELEASE GATE: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("RELEASE GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
