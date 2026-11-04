"""PostgreSQL schema and connection helpers."""
import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    service TEXT NOT NULL,
    alert TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'investigating', 'resolved')),
    relevant_knowledge TEXT NOT NULL,
    retrieval_matches JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieval_ms DOUBLE PRECISION,
    analysis_source TEXT NOT NULL,
    ai_analysis JSONB NOT NULL
);
"""

@contextmanager
def connection():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL must be configured; see .env.example")
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=5) as conn:
        yield conn

def init_db():
    with connection() as conn:
        conn.execute(SCHEMA)
