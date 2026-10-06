"""The release gate must fail closed."""

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import release_gate  # noqa: E402

GOOD = {
    "required_scanners": ["bandit", "trivy-image"],
    "blocked_with_ci_coverage": ["trivy-image"],
    "scanners": {"bandit": {"status": "PASS"}, "trivy-image": {"status": "BLOCKED"}},
    "secrets": {"found": 0},
    "tests": {"total": 10, "failed": 0},
    "dependency_vulnerabilities": 0,
    "container_blockers": 0,
    "findings": [
        {"id": "F-1", "severity": "high", "status": "fixed", "regression_test": "tests/x.py::t"},
        {"id": "F-2", "severity": "low", "status": "accepted", "acceptance": {"rationale": "r", "owner": "o", "review_date": "2999-01-01"}},
    ],
}


def run(mutate):
    data = copy.deepcopy(GOOD)
    mutate(data)
    return release_gate.evaluate(data, today=date(2026, 10, 5))


def test_a_clean_result_passes():
    assert run(lambda d: None) == []


@pytest.mark.parametrize("severity", ["critical", "high"])
def test_an_unresolved_critical_or_high_blocks(severity):
    assert run(lambda d: d["findings"].append({"id": "F-9", "severity": severity, "status": "open"}))


def test_an_auth_finding_blocks_even_if_low():
    assert run(lambda d: d["findings"].append({"id": "F-9", "severity": "low", "status": "open", "tags": ["authorization"]}))


def test_a_medium_must_be_fixed_or_properly_accepted():
    assert run(lambda d: d["findings"].append({"id": "F-9", "severity": "medium", "status": "open"}))
    assert run(lambda d: d["findings"].append({"id": "F-9", "severity": "medium", "status": "accepted", "acceptance": {"rationale": "x"}}))
    assert run(lambda d: d["findings"].append({"id": "F-9", "severity": "medium", "status": "accepted", "acceptance": {"rationale": "x", "owner": "o", "review_date": "2020-01-01"}}))


def test_a_fixed_high_without_a_regression_test_blocks():
    assert run(lambda d: d["findings"].append({"id": "F-9", "severity": "high", "status": "fixed"}))


def test_a_secret_or_a_missing_secret_result_blocks():
    assert run(lambda d: d["secrets"].update(found=1))
    assert run(lambda d: d.pop("secrets"))


@pytest.mark.parametrize("status", ["FAIL", "ERROR", "BLOCKED", "MISSING"])
def test_a_required_scanner_that_did_not_pass_blocks(status):
    def mutate(d):
        if status == "MISSING":
            d["scanners"].pop("bandit")
        else:
            d["scanners"]["bandit"]["status"] = status

    assert run(mutate)


def test_blocked_is_acceptable_only_where_ci_covers_the_scanner():
    assert run(lambda d: d["blocked_with_ci_coverage"].clear())


def test_failing_or_missing_tests_block():
    assert run(lambda d: d["tests"].update(failed=1))
    assert run(lambda d: d.pop("tests"))


def test_dependency_and_container_blockers_block():
    assert run(lambda d: d.update(dependency_vulnerabilities=1))
    assert run(lambda d: d.update(container_blockers=2))
    assert run(lambda d: d.pop("dependency_vulnerabilities"))


def test_the_script_exits_nonzero_on_unreadable_input(tmp_path):
    assert release_gate.main(["x", str(tmp_path / "missing.json")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert release_gate.main(["x", str(bad)]) == 2


def test_the_committed_results_file_is_valid_json_with_the_required_shape():
    data = json.loads((ROOT / "security-results.json").read_text(encoding="utf-8"))
    for key in ("commit", "generated", "scanners", "findings", "secrets", "tests", "required_scanners"):
        assert key in data
