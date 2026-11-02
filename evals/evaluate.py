"""Offline ranking evaluation; semantic mode requires a configured, indexed database."""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_tools import search_incident_knowledge


def evaluate_cases(cases, knowledge, retriever=search_incident_knowledge, k=3):
    if not 1 <= k <= 10 or not cases:
        raise ValueError("Expected nonempty cases and k between 1 and 10")
    seen = set()
    rows = []
    for case in cases:
        if case["id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["id"])
        labels = case.get("expected_phrases", [case.get("expected_phrase")])
        if not labels or (case["category"] != "negative" and not all(isinstance(x, str) and x.strip() for x in labels)):
            raise ValueError("Invalid expected labels")
        result = retriever(case["query"], knowledge)
        matches = [m["excerpt"] for m in result["matches"][:k]]
        ranks = [i for i, item in enumerate(matches, 1) if any(label and label.casefold() in item.casefold() for label in labels)]
        rank = min(ranks) if ranks else None
        negative = case["category"] == "negative"
        rows.append({"id": case["id"], "category": case["category"], "rank": rank,
                     "hit_at_1": rank == 1, "hit_at_3": rank is not None and rank <= 3,
                     "reciprocal_rank": 1 / rank if rank else 0,
                     "returned_results": len(matches),
                     "correct_abstention": not matches if negative else None})
    positive = [r for r in rows if r["category"] != "negative"]
    def metrics(group):
        return {"count": len(group),
                "hit_at_1": sum(x["hit_at_1"] for x in group) / len(group) if group else None,
                "hit_at_3": sum(x["hit_at_3"] for x in group) / len(group) if group else None,
                "mrr_at_3": sum(x["reciprocal_rank"] for x in group) / len(group) if group else None}
    by_category = {cat: metrics([r for r in positive if r["category"] == cat]) for cat in sorted({r["category"] for r in positive})}
    negatives = [r for r in rows if r["category"] == "negative"]
    return {"cases": len(rows), "retrieval_backend": "keyword" if retriever is search_incident_knowledge else "injected",
            "hit_at_1": metrics(positive)["hit_at_1"], "hit_at_3": metrics(positive)["hit_at_3"],
            "hit_at_k": sum(x["rank"] is not None and x["rank"] <= k for x in positive) / len(positive) if positive else None,
            "mrr_at_k": sum(x["reciprocal_rank"] for x in positive) / len(positive) if positive else None,
            "mrr_at_3": metrics(positive)["mrr_at_3"],
            "negative_abstention_rate": sum(r["correct_abstention"] for r in negatives) / len(negatives) if negatives else None,
            "by_category": by_category, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["keyword", "semantic", "both"], default="keyword")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text())
    knowledge = (ROOT / "knowledge/incidents.txt").read_text()
    reports = {}
    if args.backend in ("keyword", "both"):
        reports["keyword"] = evaluate_cases(cases, knowledge)
    if args.backend in ("semantic", "both"):
        if not os.getenv("DATABASE_URL") or not os.getenv("GEMINI_API_KEY"):
            raise SystemExit("Semantic evaluation requires DATABASE_URL, GEMINI_API_KEY and an indexed corpus")
        from vector_store import search
        def semantic(query, _):
            return {"matches": [{"excerpt": row["content"]} for row in search(query, top_k=3)]}
        reports["semantic"] = evaluate_cases(cases, knowledge, retriever=semantic)
    print(json.dumps(reports, indent=2))
