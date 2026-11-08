"""Offline checks for frozen evaluation protocol. No Gemini credentials needed."""
import json
from pathlib import Path

import pytest

from evaluation_holdout import freeze, run, validate_cases, wilson


def make_cases(tmp_path, n=30):
    """Synthetic fixture for protocol tests ONLY, not benchmark evidence."""
    from retrieval import parse_knowledge
    title = parse_knowledge()[0]["title"]
    rows = [
        {"id": f"pos-{i}", "query": f"fixture positive wording number {i} different context", "expected_title": title}
        for i in range(n)
    ] + [
        {"id": f"neg-{i}", "query": f"fixture unrelated question number {i} different context", "expected_title": None}
        for i in range(n)
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(rows))
    return path


def test_missing_and_duplicate_cases_rejected(tmp_path):
    cases = make_cases(tmp_path)
    assert len(validate_cases(cases)) == 60
    rows = json.loads(cases.read_text())
    rows[1]["query"] = rows[0]["query"]
    cases.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(cases)


def test_previous_queries_blocked(tmp_path):
    cases = make_cases(tmp_path)
    rows = json.loads(cases.read_text())
    prior = json.loads(Path("evaluation/cases.json").read_text())[0]
    rows[0]["query"] = prior["query"]
    cases.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="previously inspected"):
        validate_cases(cases)


def test_insufficient_holdout_rejected(tmp_path):
    cases = make_cases(tmp_path, 3)
    with pytest.raises(ValueError, match="at least"):
        validate_cases(cases)


def test_manifest_is_immutable_and_detects_case_edits(tmp_path):
    cases = make_cases(tmp_path)
    lock = tmp_path / "freeze.json"
    output = tmp_path / "results.json"
    freeze(cases, lock, .65821, 3.93992)
    with pytest.raises(FileExistsError):
        freeze(cases, lock, .65821, 3.93992)
    rows = json.loads(cases.read_text())
    rows[0]["query"] += " changed"
    cases.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="changed after freezing"):
        run(cases, lock, output)


def test_wilson_is_finite_not_perfect_even_for_all_successes():
    low, high = wilson(10, 10)
    assert 0 < low < high == 1
    assert wilson(0, 10)[1] > 0


def test_results_file_refuses_overwrite(tmp_path):
    cases = make_cases(tmp_path)
    lock = tmp_path / "freeze.json"
    output = tmp_path / "results.json"
    freeze(cases, lock, .65821, 3.93992)
    output.write_text("already inspected")
    with pytest.raises(FileExistsError):
        run(cases, lock, output)
