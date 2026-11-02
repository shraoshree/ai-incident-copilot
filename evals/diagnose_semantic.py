"""Inspect raw semantic ranks and cosine scores without any similarity cutoff."""
import json
from pathlib import Path
from evals.evaluate import records, verify_index
from vector_store import search

root = Path(__file__).resolve().parents[1]
cases = json.loads((root / "evals/retrieval_cases.json").read_text())
corpus = records((root / "knowledge/incidents.txt").read_text())
verify_index(corpus)
rows = []
for case in cases:
    hits = search(case["query"], top_k=10, min_similarity=-1)
    expected = case.get("expected_phrases", [case.get("expected_phrase")])
    correct = next(({"rank": i, "score": hit["score"]}
                    for i, hit in enumerate(hits, 1)
                    if hit["content"].split("\n", 1)[0].removeprefix("Incident: ").strip() in expected), None)
    rows.append({"id": case["id"], "split": case["split"], "category": case["category"],
                 "correct": correct, "top_score": hits[0]["score"] if hits else None,
                 "top_record": hits[0]["content"].split("\n", 1)[0] if hits else None})
def summary(items):
    positives = [x for x in items if x["category"] != "negative"]
    negatives = [x for x in items if x["category"] == "negative"]
    return {"cases": len(items), "positives": len(positives), "negatives": len(negatives),
            "hit_at_1": sum(x["correct"] is not None and x["correct"]["rank"] == 1 for x in positives) / len(positives),
            "hit_at_3": sum(x["correct"] is not None and x["correct"]["rank"] <= 3 for x in positives) / len(positives),
            "mrr_at_3": sum(1 / x["correct"]["rank"] for x in positives if x["correct"] and x["correct"]["rank"] <= 3) / len(positives),
            "correct_scores": sorted(x["correct"]["score"] for x in positives if x["correct"]),
            "negative_top_scores": sorted(x["top_score"] for x in negatives if x["top_score"] is not None)}
report = {"corpus_records": len(corpus),
          "dev": summary([x for x in rows if x["split"] == "dev"]),
          "test": summary([x for x in rows if x["split"] == "test"]),
          "queries": rows}
path = root / "evals" / "diagnostics.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "queries"}, indent=2))
