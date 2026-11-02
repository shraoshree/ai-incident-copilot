"""Offline vector retrieval and evaluation tests."""
import importlib.util
from pathlib import Path
import pytest
from vector_store import vector_literal, retrieve_or_fallback
from agent_tools import search_incident_knowledge


def test_vector_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        vector_literal([0.1, 0.2])


def test_vector_format():
    assert vector_literal([0.5] * 768).startswith("[0.5,0.5,")


def test_retrieval_fallback_without_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = retrieve_or_fallback(
        "database connections exhausted",
        "Incident: API latency\nSymptoms: database connections exhausted",
    )
    assert result["backend"] == "keyword"
    assert result["matches"]


def test_evaluation_reports_hit_rate():
    spec = importlib.util.spec_from_file_location("retrieval_eval", Path("evals/evaluate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.evaluate_cases(
        [{"id": "case1", "query": "database connections exhausted", "expected_phrase": "API latency", "category": "direct"}],
        "Incident: API latency\nSymptoms: database connections exhausted",
    )
    assert report["cases"] == 1
    assert report["hit_at_k"] == 1.0
