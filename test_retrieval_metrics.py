"""Evaluation metric regression tests: no external API calls."""
import importlib.util
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("retrieval_eval", ROOT / "evals/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_metrics_include_rank_and_mrr():
    cases = [
        {"id": "first", "query": "one", "expected_phrase": "correct", "category": "direct"},
        {"id": "second", "query": "two", "expected_phrase": "correct", "category": "direct"},
        {"id": "missing", "query": "three", "expected_phrase": "correct", "category": "direct"},
    ]
    def retriever(query, knowledge):
        if query == "one":
            return {"matches": [{"excerpt": "correct"}]}
        if query == "two":
            return {"matches": [{"excerpt": "wrong"}, {"excerpt": "correct"}]}
        return {"matches": [{"excerpt": "wrong"}]}
    report = evaluation.evaluate_cases(cases, "", retriever=retriever)
    assert report["cases"] == 3
    assert report["hit_at_3"] == pytest.approx(2 / 3)
    assert report["mrr_at_k"] == pytest.approx(0.5)
    assert [row["rank"] for row in report["results"]] == [1, 2, None]


def test_empty_and_duplicate_cases_are_rejected():
    with pytest.raises(ValueError):
        evaluation.evaluate_cases([], "")
    repeated = [{"id": "x", "query": "a", "expected_phrase": "b", "category": "direct"}] * 2
    with pytest.raises(ValueError):
        evaluation.evaluate_cases(repeated, "")


def test_labeled_36_case_fixture_produces_report():
    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text())
    knowledge = (ROOT / "knowledge/incidents.txt").read_text()
    report = evaluation.evaluate_cases(cases, knowledge)
    assert report["cases"] == 44
    assert len({case["id"] for case in cases}) == 44
    assert all(label in knowledge for case in cases for label in case.get("expected_phrases", [case.get("expected_phrase")]) if label)
    assert {case["category"] for case in cases} == {"direct", "paraphrase", "negative", "ambiguous"}
    assert 0 <= report["hit_at_3"] <= 1
    assert 0 <= report["mrr_at_k"] <= 1
