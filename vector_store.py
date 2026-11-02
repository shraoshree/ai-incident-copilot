"""Optional PostgreSQL/pgvector retrieval; explicit opt-in via DATABASE_URL."""
import os
from contextlib import contextmanager

DIMENSIONS = 768


def enabled():
    return bool(os.getenv("DATABASE_URL"))


@contextmanager
def connection():
    import psycopg
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    with psycopg.connect(url, connect_timeout=5) as conn:
        yield conn


def initialize():
    """Requires a database user allowed to install the vector extension."""
    with connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL UNIQUE,
                embedding vector(768) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)


def embed(text, client=None):
    from google import genai
    from google.genai import types
    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
    )
    vector = response.embeddings[0].values
    if len(vector) != DIMENSIONS:
        raise ValueError("Unexpected embedding dimension")
    return vector


def vector_literal(values):
    if len(values) != DIMENSIONS:
        raise ValueError("Expected 768-dimensional vector")
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def upsert_chunks(chunks, embedder=embed):
    with connection() as conn:
        for chunk in chunks:
            if not chunk.strip():
                continue
            vector = vector_literal(embedder(chunk))
            conn.execute(
                """INSERT INTO knowledge_chunks (content, embedding)
                   VALUES (%s, %s::vector)
                   ON CONFLICT (content) DO UPDATE SET embedding = EXCLUDED.embedding""",
                (chunk.strip(), vector),
            )


def search(query, top_k=3, embedder=embed):
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    vector = vector_literal(embedder(query))
    with connection() as conn:
        rows = conn.execute(
            """SELECT content, 1 - (embedding <=> %s::vector) AS similarity
               FROM knowledge_chunks
               ORDER BY embedding <=> %s::vector LIMIT %s""",
            (vector, vector, top_k),
        ).fetchall()
    return [{"content": content, "score": float(score)} for content, score in rows]


def retrieve_or_fallback(query, knowledge_text):
    """No silent database writes; fall back to local keyword search on outage."""
    from agent_tools import search_incident_knowledge
    if enabled():
        try:
            return {"backend": "pgvector", "matches": search(query)}
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Vector retrieval failed; using local knowledge")
    result = search_incident_knowledge(query, knowledge_text)
    return {"backend": "keyword", "matches": [
        {"content": item["excerpt"], "score": item["score"]}
        for item in result["matches"]
    ]}
