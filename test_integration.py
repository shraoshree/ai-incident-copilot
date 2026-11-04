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
