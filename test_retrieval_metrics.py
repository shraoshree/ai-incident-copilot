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
            return {"matches": [{"excerpt": "Incident: correct\nSymptoms: sample"}]}
        if query == "two":
            return {"matches": [{"excerpt": "Incident: wrong\nSymptoms: sample"}, {"excerpt": "Incident: correct\nSymptoms: sample"}]}
        return {"matches": [{"excerpt": "Incident: wrong\nSymptoms: sample"}]}
    report = evaluation.evaluate_cases(cases, "Incident: correct\nSymptoms: sample\n\nIncident: wrong\nSymptoms: sample", retriever=retriever)
    assert report["cases"] == 3
    assert report["hit_at_k"] == pytest.approx(2 / 3)
    assert report["mrr_at_k"] == pytest.approx(0.5)
    assert [row["rank"] for row in report["results"]] == [1, 2, None]


def test_empty_and_duplicate_cases_are_rejected():
    with pytest.raises(ValueError):
        evaluation.evaluate_cases([], "")
    repeated = [{"id": "x", "query": "a", "expected_phrase": "b", "category": "direct"}] * 2
    with pytest.raises(ValueError):
        evaluation.evaluate_cases(repeated, "")


def test_labeled_44_case_fixture_produces_report():
    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text())
    knowledge = (ROOT / "knowledge/incidents.txt").read_text()
    report = evaluation.evaluate_cases(cases, knowledge)
    assert report["cases"] == 44
    assert len({case["id"] for case in cases}) == 44
    assert all(label in knowledge for case in cases for label in case.get("expected_phrases", [case.get("expected_phrase")]) if label)
    assert {case["category"] for case in cases} == {"direct", "paraphrase", "negative", "ambiguous"}
    assert 0 <= report["hit_at_k"] <= 1
    assert 0 <= report["mrr_at_k"] <= 1


def test_k_one_does_not_claim_hit_at_three():
    cases = [{"id": "x", "query": "x", "category": "direct", "expected_phrase": "A"}]
    def retriever(query, knowledge):
        return {"matches": [{"excerpt": "Incident: B"}, {"excerpt": "Incident: A"}]}
    report = evaluation.evaluate_cases(cases, "Incident: A\n\nIncident: B", retriever=retriever, k=1)
    assert report["hit_at_k"] == 0
    assert report["k"] == 1
    assert "hit_at_3" not in report


def test_negative_empty_labels_and_abstention():
    cases = [{"id": "none", "query": "unrelated", "category": "negative", "expected_phrases": []}]
    report = evaluation.evaluate_cases(cases, "Incident: A", retriever=lambda q, k: {"matches": []})
    assert report["negative_abstention_rate"] == 1


def test_rejects_unrecognized_incident_chunk():
    cases = [{"id": "x", "query": "x", "category": "direct", "expected_phrase": "A"}]
    with pytest.raises(ValueError):
        evaluation.evaluate_cases(cases, "Incident: A", retriever=lambda q, k: {"matches": [{"excerpt": "Incident: OTHER"}]})


def test_all_misses_and_negative_nonabstention():
    cases = [
        {"id": "miss", "query": "a", "category": "direct", "expected_phrase": "A"},
        {"id": "false_positive", "query": "b", "category": "negative", "expected_phrases": []},
    ]
    report = evaluation.evaluate_cases(cases, "Incident: A\n\nIncident: B", retriever=lambda q, k: {"matches": [{"excerpt": "Incident: B"}]})
    assert report["hit_at_1"] == 0
    assert report["mrr_at_k"] == 0
    assert report["negative_abstention_rate"] == 0
