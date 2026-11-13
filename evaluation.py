"""Benchmark labeled semantic retrieval against original keyword baseline.

Do not equate these retrieval metrics with LLM root-cause accuracy.
"""
import json
import argparse
from pathlib import Path
from retrieval import effective_threshold, incident_query, keyword_search, semantic_search
from incident_input import Incident, prepare_incident

def case_query(case: dict) -> str:
    """Use API validation, redaction and query formatting for every method.

    Old query-only fixtures have no service/alert metadata. Preserve their
    text as description with explicit placeholders, rather than infer facts.
    New cases should supply a full API payload in the incident field.
    """
    payload = case["incident"] if "incident" in case else {
        "service": "unspecified", "alert": "Incident reported",
        "severity": "unspecified", "description": case["query"],
    }
    incident = prepare_incident(Incident.model_validate(payload))
    return incident_query(incident.service, incident.alert, incident.description)


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
    cutoff = effective_threshold() if method == "semantic" else None
    for case in cases:
        query = case_query(case)
        if method == "semantic":
            hits = semantic_search(query, top_k=3, threshold=cutoff)
        else:
            hits = keyword_search(query)
        titles = [hit["title"] for hit in hits]
        predictions.append((titles, case["expected_title"]))
        print(json.dumps({"query": query, "predicted": titles, "expected": case["expected_title"]}))
    summary = metrics(predictions)
    metadata = {"mode": "production", "threshold": cutoff} if method == "semantic" else {}
    print(json.dumps({"method": method, **metadata, **summary}, indent=2))
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/cases.json")
    parser.add_argument("--method", choices=["semantic", "keyword"], required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.method)
