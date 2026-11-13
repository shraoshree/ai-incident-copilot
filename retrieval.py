"""Embedding-based incident retrieval, with a keyword baseline for evaluations."""
import hashlib
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
import numpy as np
from pgvector.psycopg import register_vector

from database import connection

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
VECTOR_DIM = 768
DEFAULT_THRESHOLD = 0.65821  # calibrated on synthetic development queries, not production-validated

def embed(text: str, task_type: str) -> list[float]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY must be configured")
    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=VECTOR_DIM,
        ),
    )
    if not result.embeddings or not result.embeddings[0].values:
        raise RuntimeError("Embedding API returned no vector")
    values = result.embeddings[0].values
    if len(values) != VECTOR_DIM:
        raise RuntimeError(f"Expected {VECTOR_DIM} dimensions, got {len(values)}")
    return values

def parse_knowledge(path: str = "knowledge/incidents.txt") -> list[dict]:
    blocks = re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8").strip())
    records = []
    for block in blocks:
        lines = block.splitlines()
        if not lines or not lines[0].startswith("Incident: "):
            continue
        title = lines[0].removeprefix("Incident: ").strip()
        identifier = hashlib.sha256(title.lower().encode()).hexdigest()[:16]
        records.append({"id": identifier, "title": title, "content": block.strip()})
    return records

def ingest(path: str = "knowledge/incidents.txt") -> int:
    records = parse_knowledge(path)
    with connection() as conn:
        register_vector(conn)
        for record in records:
            vector = embed(record["content"], "RETRIEVAL_DOCUMENT")
            conn.execute(
                """INSERT INTO knowledge (id, title, content, embedding)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                     title=EXCLUDED.title, content=EXCLUDED.content,
                     embedding=EXCLUDED.embedding, updated_at=now()""",
                (record["id"], record["title"], record["content"], np.asarray(vector, dtype=np.float32)),
            )
    return len(records)

def incident_query(service: str, alert: str, description: str) -> str:
    return f"Service: {service}\nAlert: {alert}\nDescription: {description}"

def effective_threshold(threshold: float | None = None) -> float:
    """Resolve the API cutoff, including its environment override."""
    return float(os.getenv("MIN_SIMILARITY", str(DEFAULT_THRESHOLD))) if threshold is None else float(threshold)


def filter_matches(matches: list[dict], threshold: float, score_key: str = "similarity") -> list[dict]:
    """Filter full-precision scores inclusively, preserving retrieval order."""
    return [match for match in matches if float(match[score_key]) >= threshold]


def semantic_candidates(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve ranked candidates with full-precision scores for cutoff decisions."""
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    vector = embed(query, "RETRIEVAL_QUERY")
    with connection() as conn:
        register_vector(conn)
        rows = conn.execute(
            """SELECT id, title, content,
                      (1 - (embedding <=> %s)) AS similarity
               FROM knowledge
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (np.asarray(vector, dtype=np.float32), np.asarray(vector, dtype=np.float32), top_k),
        ).fetchall()
    return [
        {"id": r["id"], "title": r["title"], "content": r["content"],
         "similarity": float(r["similarity"])}
        for r in rows
    ]

def semantic_search(query: str, top_k: int = 3, threshold: float | None = None) -> list[dict]:
    cutoff = effective_threshold(threshold)
    matches = filter_matches(semantic_candidates(query, top_k), cutoff)
    # Rounding is only for API presentation, after the acceptance decision.
    return [{**match, "similarity": round(match["similarity"], 5)} for match in matches]


def keyword_search(query: str, records: list[dict] | None = None, top_k: int = 3) -> list[dict]:
    """Original substring-matching baseline: first matching record, no semantic ranking."""
    records = records if records is not None else parse_knowledge()
    words = [w for w in query.lower().split() if len(w) > 4]
    for record in records:
        if any(word in record["content"].lower() for word in words):
            return [record]
    return []

def retrieve(service: str, alert: str, description: str) -> tuple[list[dict], float]:
    start = time.perf_counter()
    matches = semantic_search(incident_query(service, alert, description))
    return matches, round((time.perf_counter() - start) * 1000, 2)

def context_from_matches(matches: list[dict]) -> str:
    if not matches:
        return "No sufficiently similar historical incident found. Do not assume a root cause."
    return "\n\n".join(
        f"[Historical incident {i}; similarity={m['similarity']:.3f}]\n{m['content']}"
        for i, m in enumerate(matches, 1)
    )

if __name__ == "__main__":
    import argparse
    from database import init_db
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "ingest"])
    args = parser.parse_args()
    init_db()
    if args.command == "ingest":
        print(f"Ingested {ingest()} historical incidents")
    else:
        print("Database initialized")
