"""Unit tests for credible offline retrieval evaluation."""
import pytest

from evaluation_robust import (
    BM25, apply_threshold, choose_threshold, distribution, load_cases,
    report_metrics, summarize
)
from retrieval import parse_knowledge


def test_dataset_has_twelve_no_answer_cases_and_both_splits():
    cases = load_cases("evaluation/cases.json", "evaluation/negative_cases.json")
    assert len(cases) == 33
    assert len([x for x in cases if x["expected_title"] is None]) == 12
    assert len([x for x in cases if x["split"] == "dev"]) == 17
    assert len([x for x in cases if x["split"] == "test"]) == 16


def test_bm25_ranks_relevant_documents_without_api():
    docs = parse_knowledge()
    ranker = BM25(docs)
    result = ranker.search("kafka consumer group lag behind producer")
    assert result[0]["title"] == "Kafka Consumer Lag"
    assert result[0]["score"] > 0
    assert ranker.search("zzzzzzzzzzzzzz") == []


def test_calibration_uses_dev_rows_only():
    dev = [
        {"split": "dev", "expected_title": "a", "hits": [{"title": "a", "score": 0.9}], "top1_score": 0.9, "correct_score": 0.9},
        {"split": "dev", "expected_title": "b", "hits": [{"title": "b", "score": 0.8}], "top1_score": 0.8, "correct_score": 0.8},
        {"split": "dev", "expected_title": None, "hits": [{"title": "a", "score": 0.2}], "top1_score": 0.2, "correct_score": None},
        {"split": "dev", "expected_title": None, "hits": [{"title": "a", "score": 0.1}], "top1_score": 0.1, "correct_score": None},
    ]
    threshold, metrics = choose_threshold(dev)
    assert threshold == pytest.approx(0.5)
    assert metrics["balanced_decision_accuracy"] == 1.0
    assert report_metrics(dev, threshold)["negative_abstention_rate"] == 1.0
    test = [{"query": "test positive", "split": "test", "expected_title": "a", "hits": [], "top1_score": 0.0, "correct_score": None},
            {"query": "test negative", "split": "test", "expected_title": None, "hits": [], "top1_score": 0.0, "correct_score": None}]
    assert summarize(dev + test, "bm25")["threshold"] == pytest.approx(threshold)


def test_midpoint_uses_only_dev_not_test():
    from evaluation_robust import development_margin
    rows = [
        {"expected_title": "a", "hits": [{"title": "a", "score": .8}], "top1_score": .8},
        {"expected_title": None, "hits": [{"title": "b", "score": .4}], "top1_score": .4},
    ]
    assert development_margin(rows)["midpoint"] == pytest.approx(.6)
    assert choose_threshold(rows)[0] == pytest.approx(.6)


def test_abstention_and_metric_distinguish_no_answer():
    rows = [
        {"expected_title": "a", "hits": [{"title": "x", "score": 0.9}, {"title": "a", "score": 0.8}]},
        {"expected_title": None, "hits": [{"title": "x", "score": 0.2}]},
    ]
    metrics = report_metrics(rows, .5)
    assert metrics["hit_at_1"] == 0
    assert metrics["hit_at_3"] == 1
    assert metrics["mrr_at_3"] == .5
    assert metrics["negative_abstention_rate"] == 1
    assert apply_threshold(rows[1], .5) == []


def test_score_distribution():
    assert distribution([]) == {"count": 0}
    assert distribution([.1, .2, .3])["median"] == .2


@pytest.mark.parametrize("cutoff", [.65821, .8])
def test_shared_filter_uses_full_precision_and_inclusive_boundary(cutoff):
    from retrieval import filter_matches
    hits = [{"title": "above", "score": cutoff + .000001},
            {"title": "equal", "score": cutoff},
            {"title": "below", "score": cutoff - .000001}]
    assert round(hits[-1]["score"], 5) == cutoff
    expected = hits[:2]
    assert filter_matches(hits, cutoff, score_key="score") == expected
    assert apply_threshold({"hits": hits}, cutoff) == expected


def test_production_report_uses_api_override_without_calibration(monkeypatch):
    import evaluation_robust
    monkeypatch.setenv("MIN_SIMILARITY", "0.800000123")
    monkeypatch.setattr(evaluation_robust, "choose_threshold",
                        lambda rows: pytest.fail("Production must not calibrate"))
    rows = [{"query": "q", "split": split, "expected_title": label,
             "hits": [{"title": "a", "score": .8}], "top1_score": .8,
             "correct_score": .8 if label else None}
            for split in ("dev", "test") for label in ("a", None)]
    report = summarize(rows, "semantic")
    assert report["threshold"] == .800000123
    assert report["mode"] == "production"
    assert report["test"]["positive_coverage"] == 0
    assert report["test"]["negative_abstention_rate"] == 1
    assert all(not example["accepted"] for example in report["test_examples"])
    assert report["dev_margin"] is None


def test_semantic_calibration_is_explicit_and_ignores_api_override(monkeypatch):
    monkeypatch.setenv("MIN_SIMILARITY", ".99")
    rows = [{"query": "q", "split": split, "expected_title": label,
             "hits": [{"title": "a", "score": score}], "top1_score": score,
             "correct_score": score if label else None}
            for split in ("dev", "test") for label, score in (("a", .8), (None, .2))]
    report = summarize(rows, "semantic", mode="calibration")
    assert report["mode"] == "calibration"
    assert report["threshold"] == pytest.approx(.5)
    with pytest.raises(ValueError, match="only available"):
        summarize(rows, "bm25", mode="production")
