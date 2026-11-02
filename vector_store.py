"""Optional PostgreSQL/pgvector retrieval; explicit opt-in via DATABASE_URL."""
import os
import hashlib
import json
from pathlib import Path
import time
import re
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


def embed(text, client=None, task_type="RETRIEVAL_QUERY"):
    cache_dir = Path(os.getenv('EMBEDDING_CACHE_DIR', '.cache/embeddings'))
    model = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
    key = hashlib.sha256(json.dumps([model, DIMENSIONS, task_type, text]).encode('utf-8')).hexdigest()
    cached = cache_dir / (key + '.json')
    if cached.exists():
        values = json.loads(cached.read_text(encoding='utf-8'))
        if len(values) == DIMENSIONS:
            return values

    from google import genai
    from google.genai import types
    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = None
    for attempt in range(5):
        try:
            response = client.models.embed_content(
                model=model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS, task_type=task_type),
            )
            break
        except Exception as exc:
            message = str(exc)
            if ("429" not in message and "RESOURCE_EXHAUSTED" not in message) or attempt == 4:
                raise
            retry = re.search(r"retry in ([0-9.]+)s", message, flags=re.I)
            delay = min(90.0, max(5.0, float(retry.group(1)) + 2 if retry else 15.0 * (attempt + 1)))
            time.sleep(delay)
    vector = response.embeddings[0].values
    if len(vector) != DIMENSIONS:
        raise ValueError("Unexpected embedding dimension")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp = cached.with_suffix(".tmp")
    temp.write_text(json.dumps(vector), encoding="utf-8")
    temp.replace(cached)
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
            vector = vector_literal(embedder(chunk, task_type="RETRIEVAL_DOCUMENT") if embedder is embed else embedder(chunk))
            conn.execute(
                """INSERT INTO knowledge_chunks (content, embedding)
                   VALUES (%s, %s::vector)
                   ON CONFLICT (content) DO UPDATE SET embedding = EXCLUDED.embedding""",
                (chunk.strip(), vector),
            )


def similarity_threshold():
    value = os.getenv("MIN_SIMILARITY")
    if value is None or not value.strip():
        return None
    threshold = float(value)
    if not -1 <= threshold <= 1:
        raise ValueError("MIN_SIMILARITY must be between -1 and 1")
    return threshold


def search(query, top_k=3, embedder=embed, min_similarity=None):
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
    threshold = similarity_threshold() if min_similarity is None else min_similarity
    return [{"content": content, "score": float(score)} for content, score in rows
            if threshold is None or float(score) >= threshold]


def retrieve_or_fallback(query, knowledge_text):
    """No silent database writes; fall back to local keyword search on outage."""
    if enabled():
        try:
            return {"backend": "pgvector", "matches": search(query)}
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Vector retrieval failed; using local knowledge")
    terms = {word.strip(".,:;!?()").lower() for word in query.split() if len(word) > 4}
    blocks = [part.strip() for part in knowledge_text.split("\n\n") if part.strip()]
    ranked = sorted(
        ((sum(term in block.lower() for term in terms), block) for block in blocks),
        key=lambda item: item[0],
        reverse=True,
    )
    return {"backend": "keyword", "matches": [
        {"content": block[:1500], "score": score}
        for score, block in ranked if score > 0
    ][:3]}
