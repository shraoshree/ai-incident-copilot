"""PostgreSQL/pgvector integration tests with deterministic 768D embeddings.

Requires TEST_DATABASE_URL pointing to a disposable pgvector-enabled database.
No model credentials or network inference calls are made.
"""
import os
import pytest
from vector_store import initialize, upsert_chunks, search, retrieve_or_fallback

pytestmark = pytest.mark.pgvector


def fake_embedding(text):
    # Orthogonal directions give deterministic cosine ranking.
    vector = [0.0] * 768
    lower = text.lower()
    if "database" in lower or "pool" in lower:
        vector[0] = 1.0
    elif "queue" in lower or "worker" in lower:
        vector[1] = 1.0
    else:
        vector[2] = 1.0
    return vector


@pytest.fixture
def db(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not configured")
    monkeypatch.setenv("DATABASE_URL", url)
    initialize()
    from vector_store import connection
    with connection() as conn:
        conn.execute("TRUNCATE TABLE knowledge_chunks RESTART IDENTITY")
    yield
    with connection() as conn:
        conn.execute("TRUNCATE TABLE knowledge_chunks RESTART IDENTITY")


def test_vector_search_orders_by_cosine_similarity(db):
    upsert_chunks([
        "Database pool exhausted during payments traffic",
        "Worker queue backlog after scheduler outage",
    ], embedder=fake_embedding)
    result = search("database connection pool", top_k=2, embedder=fake_embedding)
    assert len(result) == 2
    assert result[0]["content"].startswith("Database pool")
    assert result[0]["score"] > result[1]["score"]


def test_upsert_is_idempotent(db):
    upsert_chunks(["Database pool exhausted"], embedder=fake_embedding)
    upsert_chunks(["Database pool exhausted"], embedder=fake_embedding)
    from vector_store import connection
    with connection() as conn:
        count = conn.execute("SELECT count(*) FROM knowledge_chunks").fetchone()[0]
    assert count == 1


def test_vector_outage_uses_keyword_fallback(db, monkeypatch):
    import vector_store
    def fail(*args, **kwargs):
        raise RuntimeError("database temporarily unavailable")
    monkeypatch.setattr(vector_store, "search", fail)
    result = retrieve_or_fallback(
        "database connections exhausted",
        "Incident: Database pool\nSymptoms: database connections exhausted",
    )
    assert result["backend"] == "keyword"
    assert result["matches"]


def test_query_limits(db):
    with pytest.raises(ValueError):
        search("database", top_k=0, embedder=fake_embedding)
