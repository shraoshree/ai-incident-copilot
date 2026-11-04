import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from evaluation import metrics
from retrieval import context_from_matches, keyword_search, parse_knowledge

client = TestClient(main.app, headers={"X-API-Key": "test-only-copilot-secret"})

def test_home():
    assert client.get("/").json()["message"] == "AI Incident Copilot is running"

def test_input_validation():
    assert client.post("/incident", json={"service": "", "alert": "", "severity": "", "description": ""}).status_code == 422

def test_keyword_baseline():
    records = parse_knowledge()
    assert len(records) >= 10
    hits = keyword_search("payment-service high api latency", records)
    assert hits and "Payment Service" in hits[0]["title"]

def test_no_keyword_match():
    assert keyword_search("zzzzzzz noooooo abcabc", parse_knowledge()) == []

def test_retrieval_context_abstains_when_empty():
    assert "No sufficiently similar" in context_from_matches([])

def test_metrics():
    rows = [(["a", "b"], "a"), (["c", "b"], "b"), ([], "d")]
    assert metrics(rows) == {"cases": 3, "hit_at_1": 0.3333, "hit_at_3": 0.6667, "mrr_at_3": 0.5}

def test_status_transition_invalid():
    with patch.object(main, "connection") as mock_conn:
        ctx = mock_conn.return_value.__enter__.return_value
        ctx.execute.return_value.fetchone.return_value = {"status": "resolved"}
        response = client.patch("/incidents/INC-1/status", json={"status": "open"})
    assert response.status_code == 400

def test_incident_endpoint_fallback_on_retrieval_failure():
    with patch.object(main, "retrieve", side_effect=RuntimeError("offline")), \
         patch.object(main, "analyze_incident", return_value=(main.fallback("Offline"), "fallback")), \
         patch.object(main, "save_incident", return_value={"incident_id": "INC-1", "analysis_source": "fallback"}):
        response = client.post("/incident", json={
            "service": "payment-service", "alert": "latency", "severity": "critical",
            "description": "p99 rose"
        })
    assert response.status_code == 200
    assert response.json()["incident"]["analysis_source"] == "fallback"

def test_persistence_failure_not_success():
    with patch.object(main, "retrieve", return_value=([], 1.0)), \
         patch.object(main, "analyze_incident", return_value=(main.fallback("Offline"), "fallback")), \
         patch.object(main, "save_incident", side_effect=RuntimeError("db down")):
        response = client.post("/incident", json={
            "service": "payment-service", "alert": "latency", "severity": "critical",
            "description": "p99 rose"
        })
    assert response.status_code == 503
