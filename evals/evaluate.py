"""Reproducible retrieval evaluation. No model requests or credentials required."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_tools import search_incident_knowledge


def evaluate_cases(cases, knowledge, retriever=search_incident_knowledge, k=3):
    """Compute hit@k and reciprocal rank; do not interpret tiny fixtures as production quality."""
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10")
    if not cases:
        raise ValueError("At least one labeled case is required")
    outcomes = []
    seen_ids = set()
    for case in cases:
        case_id = case["id"]
        expected = case["expected_phrase"].strip().casefold()
        if not expected or case_id in seen_ids:
            raise ValueError("Each case needs a unique id and nonempty expected_phrase")
        seen_ids.add(case_id)
        result = retriever(case["query"], knowledge)
        matches = [m["excerpt"] for m in result["matches"][:k]]
        rank = next((i for i, item in enumerate(matches, 1) if expected in item.casefold()), None)
        outcomes.append({
            "id": case_id,
            "hit_at_k": rank is not None,
            "reciprocal_rank": 1.0 / rank if rank else 0.0,
            "rank": rank,
            "retrieved": len(matches),
        })
    return {
        "cases": len(outcomes),
        "k": k,
        "retrieval_backend": "keyword" if retriever is search_incident_knowledge else "injected",
        "hit_at_3": sum(row["hit_at_k"] for row in outcomes) / len(outcomes) if k == 3 else None,
        "hit_at_k": sum(row["hit_at_k"] for row in outcomes) / len(outcomes),
        "mrr_at_k": sum(row["reciprocal_rank"] for row in outcomes) / len(outcomes),
        "results": outcomes,
    }


if __name__ == "__main__":
    cases = json.loads((ROOT / "evals/retrieval_cases.json").read_text(encoding="utf-8"))
    knowledge = (ROOT / "knowledge/incidents.txt").read_text(encoding="utf-8")
    print(json.dumps(evaluate_cases(cases, knowledge), indent=2))
