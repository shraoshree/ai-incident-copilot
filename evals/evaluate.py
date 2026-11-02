"""Deterministic retrieval evaluation: no network or model credentials required."""
import json
from pathlib import Path
from agent_tools import search_incident_knowledge


def evaluate_cases(cases, knowledge, retriever=search_incident_knowledge):
    outcomes = []
    for case in cases:
        result = retriever(case["query"], knowledge)
        matches = [m["excerpt"] for m in result["matches"]]
        hit = any(case["expected_phrase"].lower() in item.lower() for item in matches)
        outcomes.append({"id": case["id"], "hit_at_3": hit, "retrieved": len(matches)})
    return {
        "cases": len(outcomes),
        "retrieval_backend": "injected" if retriever is not search_incident_knowledge else "keyword",
        "hit_at_3": sum(item["hit_at_3"] for item in outcomes) / len(outcomes) if outcomes else 0.0,
        "results": outcomes,
    }


if __name__ == "__main__":
    cases = json.loads(Path("evals/retrieval_cases.json").read_text())
    knowledge = Path("knowledge/incidents.txt").read_text()
    print(json.dumps(evaluate_cases(cases, knowledge), indent=2))
