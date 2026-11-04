"""Benchmark labeled semantic retrieval against original keyword baseline.

Do not equate these retrieval metrics with LLM root-cause accuracy.
"""
import json
import argparse
from pathlib import Path
from retrieval import keyword_search, semantic_search

def metrics(results: list[tuple[list[str], str]]) -> dict:
    if not results:
        raise ValueError("No evaluation cases")
    n = len(results)
    hit1 = sum(bool(ids) and ids[0] == expected for ids, expected in results) / n
    hit3 = sum(expected in ids[:3] for ids, expected in results) / n
    mrr3 = sum((1 / (ids[:3].index(expected) + 1)) if expected in ids[:3] else 0 for ids, expected in results) / n
    return {"cases": n, "hit_at_1": round(hit1, 4), "hit_at_3": round(hit3, 4), "mrr_at_3": round(mrr3, 4)}

def evaluate(dataset: str, method: str):
    cases = json.loads(Path(dataset).read_text(encoding="utf-8"))
    predictions = []
    for case in cases:
        if method == "semantic":
            hits = semantic_search(case["query"], top_k=3)
        else:
            hits = keyword_search(case["query"])
        titles = [hit["title"] for hit in hits]
        predictions.append((titles, case["expected_title"]))
        print(json.dumps({"query": case["query"], "predicted": titles, "expected": case["expected_title"]}))
    summary = metrics(predictions)
    print(json.dumps({"method": method, **summary}, indent=2))
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/cases.json")
    parser.add_argument("--method", choices=["semantic", "keyword"], required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.method)
