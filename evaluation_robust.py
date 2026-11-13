"""Retrieval benchmark with no-answer cases, BM25 and dev-only calibration.

Semantic reports default to the effective API cutoff. Threshold experiments
require calibration mode; BM25 defaults to calibration as it is not an API path.
Calibration uses only development rows. All cases are synthetic, and their labels
were authored for this demo; results are not an independent production estimate.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from retrieval import effective_threshold, filter_matches, parse_knowledge, semantic_candidates
from evaluation import case_query

TOKENS = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKENS.findall(text.lower())


class BM25:
    """Small pure-Python Okapi BM25 baseline. No network or paid dependency."""

    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.k1, self.b = k1, b
        self.terms = [Counter(tokenize(r["content"])) for r in records]
        self.lengths = [sum(c.values()) for c in self.terms]
        self.avg_len = sum(self.lengths) / len(records) if records else 1
        df = Counter(term for row in self.terms for term in row)
        self.idf = {
            term: math.log(1 + (len(records) - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        words = set(tokenize(query))
        candidates = []
        for record, tf, length in zip(self.records, self.terms, self.lengths):
            score = sum(
                self.idf.get(term, 0.0) * (tf[term] * (self.k1 + 1)) /
                (tf[term] + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                for term in words if tf[term]
            )
            if score > 0:
                candidates.append({
                    "title": record["title"], "score": round(score, 8)
                })
        return sorted(candidates, key=lambda x: (-x["score"], x["title"]))[:top_k]


def load_cases(positive_path: str, negative_path: str) -> list[dict]:
    positives = json.loads(Path(positive_path).read_text(encoding="utf-8"))
    negatives = json.loads(Path(negative_path).read_text(encoding="utf-8"))
    if len(negatives) < 10:
        raise ValueError("At least 10 no-answer queries are required")
    if any(item.get("expected_title") is not None for item in negatives):
        raise ValueError("No-answer queries must have null expected_title")
    if any(not item.get("expected_title") for item in positives):
        raise ValueError("Positive queries must have expected_title")
    if len({x["query"] for x in positives + negatives}) != len(positives + negatives):
        raise ValueError("Duplicate evaluation queries")
    # Each label type is split separately: balanced development and test sets.
    return [
        {**row, "split": "dev" if i % 2 == 0 else "test"}
        for group in (positives, negatives)
        for i, row in enumerate(group)
    ]


def score_rows(cases: list[dict], method: str, records: list[dict]) -> list[dict]:
    bm25 = BM25(records) if method == "bm25" else None
    rows = []
    for case in cases:
        query = case_query(case)
        if method == "bm25":
            hits = bm25.search(query, top_k=3)
        else:
            # Disabling the filter only for measurement makes the score
            # distribution visible even if the application would abstain.
            hits = [
                {"title": hit["title"], "score": float(hit["similarity"])}
                for hit in semantic_candidates(query, top_k=3)
            ]
        rows.append({
            "query": query, "split": case["split"],
            "expected_title": case["expected_title"],
            "hits": hits, "top1_score": hits[0]["score"] if hits else 0.0,
            "correct_score": next(
                (h["score"] for h in hits if h["title"] == case["expected_title"]), None
            ) if case["expected_title"] else None,
        })
    return rows


def apply_threshold(row: dict, threshold: float) -> list[dict]:
    return filter_matches(row["hits"], threshold, score_key="score")


def report_metrics(rows: list[dict], threshold: float) -> dict:
    pos = [r for r in rows if r["expected_title"]]
    neg = [r for r in rows if not r["expected_title"]]
    if not pos or not neg:
        raise ValueError("A report needs positives and no-answer cases")
    hits = [apply_threshold(r, threshold) for r in pos]
    hit1 = sum(bool(h) and h[0]["title"] == r["expected_title"] for r, h in zip(pos, hits))
    hit3 = sum(any(x["title"] == r["expected_title"] for x in h) for r, h in zip(pos, hits))
    mrr = sum(
        next((1 / i for i, h in enumerate(matches, 1)
              if h["title"] == row["expected_title"]), 0)
        for row, matches in zip(pos, hits)
    )
    accepted_pos = sum(bool(h) for h in hits)
    correct_rejects = sum(not apply_threshold(r, threshold) for r in neg)
    return {
        "positive_cases": len(pos), "negative_cases": len(neg),
        "hit_at_1": round(hit1 / len(pos), 4),
        "hit_at_3": round(hit3 / len(pos), 4),
        "mrr_at_3": round(mrr / len(pos), 4),
        "positive_coverage": round(accepted_pos / len(pos), 4),
        "negative_abstention_rate": round(correct_rejects / len(neg), 4),
        "negative_false_accept_rate": round(1 - correct_rejects / len(neg), 4),
        "balanced_decision_accuracy": round(
            (hit1 / len(pos) + correct_rejects / len(neg)) / 2, 4
        ),
    }


def development_margin(dev_rows: list[dict]) -> dict | None:
    """A margin exists only when every dev positive is top-1 correct and
    scores strictly above every dev no-answer top-1.
    Never inspect test data when selecting a threshold.
    """
    positives = [r for r in dev_rows if r["expected_title"]]
    negatives = [r for r in dev_rows if not r["expected_title"]]
    if not positives or not negatives:
        return None
    if any(not r["hits"] or r["hits"][0]["title"] != r["expected_title"] for r in positives):
        return None
    min_correct = min(r["top1_score"] for r in positives)
    max_irrelevant = max(r["top1_score"] for r in negatives)
    if max_irrelevant >= min_correct:
        return None
    return {
        "max_dev_irrelevant": max_irrelevant,
        "min_dev_correct": min_correct,
        "gap": min_correct - max_irrelevant,
        "midpoint": (min_correct + max_irrelevant) / 2,
    }


def choose_threshold(dev_rows: list[dict]) -> tuple[float, dict]:
    # Use the center of an actual dev-only separation margin, if one exists.
    margin = development_margin(dev_rows)
    if margin is not None:
        threshold = margin["midpoint"]
        return threshold, report_metrics(dev_rows, threshold)
    # Without full separation, use the dev threshold that maximizes balanced
    # decision accuracy. Select the median best candidate to avoid favoring
    # the lowest possible threshold at a tie.
    scores = sorted({r["top1_score"] for r in dev_rows} |
                    {h["score"] for r in dev_rows for h in r["hits"]})
    candidates = sorted(set([0.00000001, scores[-1] + 0.000001] +
                            [(a + b) / 2 for a, b in zip(scores, scores[1:])] +
                            scores))
    scores_by_cutoff = [(c, report_metrics(dev_rows, c)) for c in candidates]
    best_score = max(m["balanced_decision_accuracy"] for _, m in scores_by_cutoff)
    best_candidates = [c for c, m in scores_by_cutoff
                       if m["balanced_decision_accuracy"] == best_score]
    best = best_candidates[len(best_candidates) // 2]
    return best, report_metrics(dev_rows, best)

def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def percentile(p):
        return round(ordered[round((len(ordered) - 1) * p)], 4)
    return {
        "count": len(values), "min": round(ordered[0], 4),
        "p10": percentile(.1), "median": round(statistics.median(ordered), 4),
        "p90": percentile(.9), "max": round(ordered[-1], 4)
    }


def summarize(rows: list[dict], method: str, mode: str | None = None) -> dict:
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    mode = mode or ("production" if method == "semantic" else "calibration")
    if mode not in ("production", "calibration"):
        raise ValueError("Mode must be production or calibration")
    if mode == "production" and method != "semantic":
        raise ValueError("Production mode is only available for the API semantic retriever")
    if mode == "production":
        threshold = effective_threshold()
        dev_metrics = report_metrics(dev, threshold)
        selection = "effective API MIN_SIMILARITY or default; no calibration"
        margin = None
    else:
        threshold, dev_metrics = choose_threshold(dev)
        margin = development_margin(dev)
        selection = ("midpoint of full development separation margin" if margin is not None else
                     "max development balanced decision accuracy; median tie")
    return {
        "method": method,
        "benchmark": "synthetic, query-level dev/test split, not independent",
        "mode": mode,
        "threshold": threshold,
        "threshold_selection": selection,
        "dev_margin": margin,
        "dev": dev_metrics,
        "test": report_metrics(test, threshold),
        "test_scores": {
            "correct_relevant_top3": distribution([
                r["correct_score"] for r in test if r["correct_score"] is not None
            ]),
            "irrelevant_query_top1": distribution([
                r["top1_score"] for r in test if r["expected_title"] is None
            ]),
        },
        "test_examples": [
            {
                "query": r["query"], "expected_title": r["expected_title"],
                "top1": r["hits"][0]["title"] if r["hits"] else None,
                "top1_score": r["top1_score"],
                "accepted_matches": apply_threshold(r, threshold),
                "accepted": bool(apply_threshold(r, threshold)),
            }
            for r in test
        ],
    }


def evaluate(method: str, positive_path: str = "evaluation/cases.json",
             negative_path: str = "evaluation/negative_cases.json",
             mode: str | None = None) -> dict:
    records = parse_knowledge()
    cases = load_cases(positive_path, negative_path)
    valid_titles = {r["title"] for r in records}
    if any(c["expected_title"] and c["expected_title"] not in valid_titles for c in cases):
        raise ValueError("An expected title is missing from the knowledge corpus")
    rows = score_rows(cases, method, records)
    summary = summarize(rows, method, mode)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "semantic"], required=True)
    parser.add_argument("--mode", choices=["production", "calibration"],
                        help="Default: production for semantic; calibration for BM25")
    args = parser.parse_args()
    evaluate(args.method, mode=args.mode)
