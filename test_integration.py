"""Real PostgreSQL + pgvector integration tests (no external Gemini requests)."""
import os
import uuid
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pgvector.psycopg import register_vector

import main
import retrieval
from database import connection, init_db

pytestmark = pytest.mark.integration


@pytest.fixture
def db():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not configured (CI provides a PostgreSQL service)")
    init_db()
    with connection() as conn:
        conn.execute("TRUNCATE TABLE incidents, knowledge")
    yield
    with connection() as conn:
        conn.execute("TRUNCATE TABLE incidents, knowledge")


def unit_vector(axis):
    x = np.zeros(768, dtype=np.float32)
    x[axis] = 1.0
    return x


def populate():
    with connection() as conn:
        register_vector(conn)
        for ident, title, axis in [
            ("db", "Database Saturation", 0),
            ("cpu", "CPU Overload", 1),
            ("tls", "TLS Certificate Expiry", 2),
        ]:
            conn.execute(
                "INSERT INTO knowledge (id, title, content, embedding) VALUES (%s,%s,%s,%s)",
                (ident, title, f"Incident: {title}", unit_vector(axis)),
            )


def test_database_pgvector_ranking_and_threshold(db, monkeypatch):
    populate()
    monkeypatch.setattr(retrieval, "embed", lambda text, task: unit_vector(0))
    hits = retrieval.semantic_search("DB connections exhausted", top_k=3, threshold=0.7)
    assert [h["id"] for h in hits] == ["db"]
    assert hits[0]["similarity"] == pytest.approx(1)
    assert retrieval.semantic_search("anything", top_k=2, threshold=1.01) == []


def test_pgvector_ingest_upsert(db, tmp_path, monkeypatch):
    path = tmp_path / "incidents.txt"
    path.write_text("Incident: Test Outage\nSymptoms: backend outage", encoding="utf-8")
    monkeypatch.setattr(retrieval, "embed", lambda text, task: unit_vector(0))
    assert retrieval.ingest(str(path)) == 1
    assert retrieval.ingest(str(path)) == 1
    with connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM knowledge").fetchone()["n"]
    assert count == 1


def test_api_end_to_end_with_real_db_and_stub_embeddings(db, monkeypatch):
    populate()
    monkeypatch.setattr(retrieval, "embed", lambda text, task: unit_vector(0))
    monkeypatch.setattr(main, "analyze_incident", lambda incident, context: (
        main.fallback("LLM unavailable"), "fallback"
    ))
    client = TestClient(main.app, headers={"X-API-Key": "test-only-copilot-secret"})
    assert client.get("/health").status_code == 200
    payload = {
        "service": "payments", "alert": "DB pool exhaustion", "severity": "critical",
        "description": "Connections are saturated"
    }
    response = client.post("/incident", json=payload)
    assert response.status_code == 200, response.text
    row = response.json()["incident"]
    incident_id = row["incident_id"]
    assert row["retrieval_matches"][0]["id"] == "db"
    assert row["analysis_source"] == "fallback"
    assert row["retrieval_ms"] >= 0

    assert client.get(f"/incidents/{incident_id}").status_code == 200
    assert client.get("/incidents?status=open").json()["count"] == 1
    assert client.get("/stats").json()["critical"] == 1
    assert client.patch(f"/incidents/{incident_id}/status",
                        json={"status": "investigating"}).status_code == 200
    assert client.patch(f"/incidents/{incident_id}/status",
                        json={"status": "resolved"}).status_code == 200
    assert client.patch(f"/incidents/{incident_id}/status",
                        json={"status": "open"}).status_code == 400
    assert client.get("/stats").json()["resolved"] == 1


@pytest.mark.parametrize("kind", ["structured", "legacy", "no_match"])
def test_api_and_evaluation_retrieval_parity(db, monkeypatch, tmp_path, capsys, kind):
    """Exercise API and both evaluators through real pgvector ranking.

    Only the external embedding/generation services are replaced. The embedding
    stub verifies the actual text received, including redaction and all fields.
    """
    import json
    from evaluation import evaluate
    from evaluation_robust import apply_threshold, score_rows

    populate()
    monkeypatch.delenv("MIN_SIMILARITY", raising=False)
    if kind == "legacy":
        payload = {"service": "unspecified", "alert": "Incident reported",
                   "severity": "unspecified", "description": "DB pool password=private-value"}
        case = {"query": payload["description"]}
        expected_query = "Service: unspecified\nAlert: Incident reported\nDescription: DB pool password=[REDACTED]"
    else:
        payload = {"service": "payments owner@example.com",
                   "alert": "DB pool token=private-value", "severity": "critical",
                   "description": "Connections saturated at 10.2.3.4"}
        case = {"incident": payload}
        expected_query = ("Service: payments [REDACTED_EMAIL]\n"
                          "Alert: DB pool token=[REDACTED]\n"
                          "Description: Connections saturated at [REDACTED_IP]")
    received_queries = []

    def embedding(query, task):
        received_queries.append(query)
        assert query == expected_query
        assert task == "RETRIEVAL_QUERY"
        return unit_vector(3 if kind == "no_match" else 0)

    monkeypatch.setattr(retrieval, "embed", embedding)
    monkeypatch.setattr(main, "analyze_incident", lambda incident, context: (
        main.fallback("Test generation stub"), "fallback"
    ))
    client = TestClient(main.app, headers={"X-API-Key": "test-only-copilot-secret"})
    response = client.post("/incident", json=payload)
    assert response.status_code == 200, response.text
    api_hits = response.json()["incident"]["retrieval_matches"]
    expected = [] if kind == "no_match" else [{"title": "Database Saturation", "score": 1.0}]
    assert [{"title": h["title"], "score": h["similarity"]} for h in api_hits] == expected

    case.update(split="test", expected_title="Database Saturation")
    rows = score_rows([case], "semantic", [])
    assert apply_threshold(rows[0], retrieval.DEFAULT_THRESHOLD) == expected
    assert rows[0]["query"] == expected_query

    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([case]))
    evaluate(str(dataset), "semantic")
    printed = capsys.readouterr().out.splitlines()
    assert json.loads(printed[0])["predicted"] == [h["title"] for h in api_hits]
    assert received_queries == [expected_query] * 3


@pytest.mark.parametrize("override", [None, "0.8"])
def test_api_evaluation_cutoff_boundaries_and_multiple_matches(db, monkeypatch, override):
    from evaluation_robust import apply_threshold, score_rows, summarize

    if override is None:
        monkeypatch.delenv("MIN_SIMILARITY", raising=False)
    else:
        monkeypatch.setenv("MIN_SIMILARITY", override)
    cutoff = retrieval.effective_threshold()
    with connection() as conn:
        register_vector(conn)
        for ident, score in [("high", .95), ("above", cutoff + .000001),
                             ("below", cutoff - .000001)]:
            vector = unit_vector(0) * score
            vector[1] = np.sqrt(1 - score * score)
            conn.execute("INSERT INTO knowledge (id,title,content,embedding) VALUES (%s,%s,%s,%s)",
                         (ident, ident, ident, vector))
    monkeypatch.setattr(retrieval, "embed", lambda query, task: unit_vector(0))
    monkeypatch.setattr(main, "analyze_incident", lambda incident, context: (
        main.fallback("Test"), "fallback"))
    payload = {"service": "payments", "alert": "slow", "severity": "critical", "description": "slow"}
    response = TestClient(main.app, headers={"X-API-Key": "test-only-copilot-secret"}).post(
        "/incident", json=payload)
    assert response.status_code == 200, response.text
    api_hits = response.json()["incident"]["retrieval_matches"]
    assert [hit["title"] for hit in api_hits] == ["high", "above"]
    cases = [{"incident": payload, "expected_title": "high", "split": "test"},
             {"incident": payload, "expected_title": None, "split": "dev"}]
    rows = score_rows(cases, "semantic", [])
    assert [hit["title"] for hit in rows[0]["hits"]] == ["high", "above", "below"]
    below = rows[0]["hits"][-1]["score"]
    assert below < cutoff
    assert round(below, 5) == cutoff  # The old rounding-before-filtering bug.
    accepted = apply_threshold(rows[0], cutoff)
    assert [{"title": h["title"], "score": round(h["score"], 5)} for h in accepted] == [
        {"title": h["title"], "score": h["similarity"]} for h in api_hits]
    # Supply both classes in each split to exercise the production report.
    report_rows = rows + [{**rows[0], "split": "dev"}, {**rows[1], "split": "test"}]
    report = summarize(report_rows, "semantic")
    assert report["mode"] == "production"
    assert report["threshold"] == cutoff
    assert report["test_examples"][0]["accepted_matches"] == accepted
