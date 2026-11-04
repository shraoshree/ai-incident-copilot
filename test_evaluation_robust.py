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
    assert 0.2 < threshold <= 0.8
    assert metrics["balanced_decision_accuracy"] == 1.0
    assert report_metrics(dev, threshold)["negative_abstention_rate"] == 1.0
    test = [{"split": "test", "expected_title": "a", "hits": [], "top1_score": 0.0, "correct_score": None},
            {"split": "test", "expected_title": None, "hits": [], "top1_score": 0.0, "correct_score": None}]
    assert summarize(dev + test, "bm25")["threshold"] == pytest.approx(threshold)


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
