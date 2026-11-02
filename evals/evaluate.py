"""Comparable retrieval benchmark; live semantic evaluation is opt-in."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_tools import search_incident_knowledge


def records(text):
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    ids = []
    for block in blocks:
        match = re.match(r"Incident: ([^\n]+)", block)
        if not match:
            raise ValueError("Every knowledge chunk must begin with an Incident title")
        ids.append(match.group(1).strip())
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate incident titles")
    return dict(zip(ids, blocks))


def evaluate_cases(cases, knowledge, retriever=search_incident_knowledge, k=3, backend=None):
    if not 1 <= k <= 10 or not cases:
        raise ValueError("Expected nonempty cases and k between 1 and 10")
    corpus = records(knowledge)
    seen = set()
    rows = []
    for case in cases:
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        negative = case["category"] == "negative"
        labels = case.get("expected_phrases", [case.get("expected_phrase")])
        if negative:
            if labels not in ([], [None]):
                raise ValueError("Negative cases must have no expected incident")
        elif not labels or any(label not in corpus for label in labels):
            raise ValueError("Expected incident IDs must exist in corpus")
        result = retriever(case["query"], knowledge)
        matches = result["matches"][:k]
        retrieved = []
        for item in matches:
            excerpt = item.get("excerpt", item.get("content", ""))
            match = re.match(r"Incident: ([^\n]+)", excerpt)
            if not match or match.group(1).strip() not in corpus:
                raise ValueError("Retriever returned a chunk without a valid incident ID")
            retrieved.append(match.group(1).strip())
        rank = next((i for i, name in enumerate(retrieved, 1) if name in labels), None)
        rows.append({"id": case["id"], "category": case["category"], "rank": rank,
                     "hit_at_1": rank == 1, "hit_at_k": rank is not None,
                     "reciprocal_rank": 1 / rank if rank else 0.0,
                     "returned_results": len(retrieved),
                     "correct_abstention": len(retrieved) == 0 if negative else None})
    positive = [row for row in rows if row["category"] != "negative"]
    def metrics(group):
        n = len(group)
        return {"count": n,
                "hit_at_1": sum(row["hit_at_1"] for row in group) / n if n else None,
                "hit_at_k": sum(row["hit_at_k"] for row in group) / n if n else None,
                "mrr_at_k": sum(row["reciprocal_rank"] for row in group) / n if n else None}
    summary = metrics(positive)
    negatives = [row for row in rows if row["category"] == "negative"]
    return {"cases": len(rows), "corpus_records": len(corpus), "k": k,
            "retrieval_backend": backend or ("keyword" if retriever is search_incident_knowledge else "injected"),
            **summary,
            "negative_abstention_rate": sum(row["correct_abstention"] for row in negatives) / len(negatives) if negatives else None,
            "by_category": {cat: metrics([row for row in positive if row["category"] == cat])
                            for cat in sorted({row["category"] for row in positive})},
            "results": rows}


def verify_index(corpus):
    from vector_store import connection
    with connection() as conn:
        indexed = {row[0] for row in conn.execute("SELECT content FROM knowledge_chunks").fetchall()}
    expected = set(corpus.values())
    if indexed != expected:
        raise RuntimeError(f"Stale semantic index: {len(indexed)} indexed vs {len(expected)} corpus records; re-run scripts/index_knowledge.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["keyword", "semantic", "both"], default="keyword")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--tune-threshold", action="store_true", help="Calibrate semantic cutoff using dev negatives and positives only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-similarity", type=float, default=None,
                        help="Calibrated cosine similarity cutoff; do not tune on held-out test cases")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between semantic embedding requests")
    args = parser.parse_args()
    if args.delay < 0 or (args.min_similarity is not None and not -1 <= args.min_similarity <= 1):
        parser.error("Invalid delay or similarity threshold")
    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text())
    knowledge = (ROOT / "knowledge/incidents.txt").read_text()
    corpus = records(knowledge)
    if args.tune_threshold:
        if args.split != "dev" or args.backend != "semantic":
            parser.error("Threshold tuning requires --split dev --backend semantic")
        if not os.getenv("DATABASE_URL") or not os.getenv("GEMINI_API_KEY"):
            parser.error("Semantic evaluation requires DATABASE_URL and GEMINI_API_KEY")
        verify_index(corpus)
        from vector_store import search
        negatives = [case for case in cases if case["split"] == "dev" and case["category"] == "negative"]
        positives = [case for case in cases if case["split"] == "dev" and case["category"] != "negative"]
        if not negatives or not positives:
            parser.error("Need both dev negative and positive examples")
        scores = {case["id"]: search(case["query"], top_k=args.k, min_similarity=-1) for case in negatives + positives}
        candidates = sorted({float(row["score"]) for rows in scores.values() for row in rows} | {1.0})
        def utility(cutoff):
            good = sum(any(row["score"] >= cutoff and row["content"].split("\\n", 1)[0][10:] in case.get("expected_phrases", [case.get("expected_phrase")]) for row in scores[case["id"]][:1]) for case in positives)
            abstain = sum(not any(row["score"] >= cutoff for row in scores[case["id"]]) for case in negatives)
            return good / len(positives) + abstain / len(negatives)
        best = max(candidates, key=lambda value: (utility(value), value))
        print(json.dumps({"dev_cases": len(positives) + len(negatives), "recommended_min_similarity": best, "dev_utility": utility(best)}, indent=2))
        return
    if args.split != "all":
        cases = [case for case in cases if case["split"] == args.split]
    reports = {}
    if args.backend in ("keyword", "both"):
        reports["keyword"] = evaluate_cases(cases, knowledge, k=args.k, backend="keyword")
    if args.backend in ("semantic", "both"):
        if not os.getenv("DATABASE_URL") or not os.getenv("GEMINI_API_KEY"):
            parser.error("Semantic evaluation requires DATABASE_URL, GEMINI_API_KEY and an indexed corpus")
        verify_index(corpus)
        from vector_store import search, similarity_threshold
        threshold = args.min_similarity if args.min_similarity is not None else similarity_threshold()
        def semantic(query, _):
            time.sleep(args.delay)
            rows = search(query, top_k=args.k, min_similarity=threshold)
            return {"matches": [{"excerpt": row["content"]} for row in rows]}
        reports["semantic"] = evaluate_cases(cases, knowledge, retriever=semantic, k=args.k, backend="semantic")
        reports["semantic"]["min_similarity"] = threshold
        reports["semantic"]["abstention_configured"] = threshold is not None
    output = json.dumps(reports, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
