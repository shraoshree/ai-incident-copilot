"""One-shot, frozen, independently authored retrieval evaluation.

This is separate from evaluation_robust.py, whose old test cases influenced
development. Do not tune the model/thresholds or rewrite cases after freeze.
"""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation_robust import report_metrics, score_rows
from retrieval import EMBED_MODEL, parse_knowledge

FROZEN_FILES = (
    "retrieval.py", "evaluation_robust.py", "knowledge/incidents.txt",
)
PREVIOUS_CASE_FILES = ("evaluation/cases.json", "evaluation/negative_cases.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(query):
    return " ".join(query.casefold().split())


def validate_cases(path, minimum=30):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Cases must be a JSON array")
    seen_ids, seen_queries = set(), set()
    previous = {
        normalized(row["query"])
        for old in PREVIOUS_CASE_FILES
        for row in json.loads(Path(old).read_text(encoding="utf-8"))
    }
    titles = {r["title"] for r in parse_knowledge()}
    pos = neg = 0
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("query"), str):
            raise ValueError("Each case needs string id and query")
        identifier, query = row["id"].strip(), normalized(row["query"])
        if not identifier or not query or identifier in seen_ids or query in seen_queries:
            raise ValueError("Blank or duplicate id/query")
        if query in previous:
            raise ValueError("A new holdout reuses a previously inspected query")
        seen_ids.add(identifier)
        seen_queries.add(query)
        if "expected_title" not in row:
            raise ValueError("Missing expected_title (null for no-answer)")
        if row["expected_title"] is None:
            neg += 1
        elif row["expected_title"] in titles:
            pos += 1
        else:
            raise ValueError("Expected title absent from knowledge corpus")
    if pos < minimum or neg < minimum:
        raise ValueError("Need at least %d positive and %d no-answer cases; got %d/%d" % (minimum, minimum, pos, neg))
    return cases


def freeze(cases_path, manifest_path, semantic_cutoff, bm25_cutoff, minimum=30):
    validate_cases(cases_path, minimum)
    out = Path(manifest_path)
    if out.exists():
        raise FileExistsError("Manifest exists; never overwrite an existing freeze")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": 1,
        "purpose": "new one-shot evaluation; prior evaluation was test-informed",
        "cases_sha256": sha(cases_path),
        "file_hashes": {p: sha(p) for p in FROZEN_FILES},
        "embedding_model": EMBED_MODEL,
        "thresholds": {"semantic": float(semantic_cutoff), "bm25": float(bm25_cutoff)},
        "minimum_each_class": int(minimum),
        "note": "Thresholds were fixed before this holdout was evaluated. Do not re-tune on these results.",
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def wilson(successes, total, z=1.96):
    """95% Wilson interval; descriptive only, not proof of generalization."""
    if total == 0:
        raise ValueError("Cannot compute interval for empty class")
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** .5) / den
    return [round(max(0, center - half), 4), round(min(1, center + half), 4)]


def run(cases_path, manifest_path, output_path):
    out = Path(output_path)
    if out.exists():
        raise FileExistsError("Result exists: refusing to rerun or overwrite holdout")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cases = validate_cases(cases_path, manifest["minimum_each_class"])
    if sha(cases_path) != manifest["cases_sha256"]:
        raise ValueError("Holdout cases changed after freezing")
    if {p: sha(p) for p in FROZEN_FILES} != manifest["file_hashes"]:
        raise ValueError("Retrieval/evaluation source or corpus changed after freezing")
    if EMBED_MODEL != manifest["embedding_model"]:
        raise ValueError("Embedding model changed after freezing")
    records = parse_knowledge()
    results = {}
    for method in ("bm25", "semantic"):
        # No development split or threshold selection on the new holdout.
        rows = score_rows([{**c, "split": "test"} for c in cases], method, records)
        cutoff = manifest["thresholds"][method]
        metrics = report_metrics(rows, cutoff)
        p, n = metrics["positive_cases"], metrics["negative_cases"]
        metrics["hit_at_1_95pct_wilson"] = wilson(round(metrics["hit_at_1"] * p), p)
        metrics["negative_abstention_95pct_wilson"] = wilson(round(metrics["negative_abstention_rate"] * n), n)
        results[method] = {"threshold": cutoff, "metrics": metrics, "rows": rows}
    report = {
        "status": "one-shot synthetic holdout; not production accuracy",
        "manifest_sha256": sha(manifest_path),
        "cases_sha256": sha(cases_path),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents quietly rewriting previously seen results.
    with out.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze", "run"])
    ap.add_argument("--cases", default="evaluation/fresh_holdout.json")
    ap.add_argument("--manifest", default="evaluation/fresh_holdout.lock.json")
    ap.add_argument("--output", default="evaluation/fresh_holdout_results.json")
    ap.add_argument("--semantic-threshold", type=float, default=0.65821)
    ap.add_argument("--bm25-threshold", type=float, default=3.93992)
    ap.add_argument("--minimum-each-class", type=int, default=30)
    args = ap.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.cases, args.manifest, args.semantic_threshold, args.bm25_threshold, args.minimum_each_class), indent=2))
    else:
        summary = run(args.cases, args.manifest, args.output)
        print(json.dumps({k: v["metrics"] for k, v in summary["results"].items()}, indent=2))
