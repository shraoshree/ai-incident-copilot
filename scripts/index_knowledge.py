"""Run explicitly to index historical knowledge into pgvector."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vector_store import initialize, upsert_chunks

if __name__ == "__main__":
    text = Path("knowledge/incidents.txt").read_text(encoding="utf-8")
    chunks = [part.strip() for part in text.split("\n\n") if part.strip()]
    initialize()
    upsert_chunks(chunks)
    print(f"Indexed {len(chunks)} knowledge chunks")
