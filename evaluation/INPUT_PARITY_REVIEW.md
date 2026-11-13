# Evaluation input parity review

Scope: match API input processing before investigating root-cause confusion.
Branch: `fix/evaluation-api-input-parity`. PR #6 was reviewed, not merged or
cherry-picked. Its source/corpus hash and historical-result preservation approach
was reused for `reference-query-input/manifest.json`; its independent holdout
framework remains outside this change.

## Reference saved before changes

Base commit: `b4a3184a73a24e0a058ea961529b648b145842da`.
Local tag: `reference/before-evaluation-api-input-parity`.
`reference-query-input/` contains source/data SHA-256 hashes, the original keyword
and BM25 outputs, dependency versions, and baseline test output. The original
November 4 semantic results and datasets are unchanged. No new live semantic
baseline was obtainable: this environment has no Gemini credential.

## Changes

- Move the existing Incident model and redaction operation into `incident_input.py`.
  The API imports them; its validation, redaction, and search behavior stay the same.
- Both evaluators validate and redact an Incident, then call the same
  `retrieval.incident_query` formatter the API uses. All retrieval methods receive
  the formatted input; reports show the actual redacted search query.
- New cases can contain a full `incident` payload, for example:

```json
{
  "incident": {
    "service": "payments",
    "alert": "Checkout latency",
    "severity": "critical",
    "description": "p99 latency reached 2800ms"
  },
  "expected_title": "Payment Service High Latency"
}
```

Existing query-only cases are explicitly adapted to `service="unspecified"`,
`alert="Incident reported"`, `severity="unspecified"`, and `description=query`.
These placeholder-wrapped synthetic cases test the API input shape but cannot
establish performance with realistic service metadata. Existing splits, labels,
corpus, semantic cutoff, generation prompts and threshold-selection code are
unchanged. The robust evaluator still temporarily disables filtering to measure
candidate scores, then applies its existing evaluation cutoff; the parity test
applies the API cutoff when comparing its results with the API.

## Actual validation

Python 3.12; isolated PostgreSQL/pgvector 16 database. Commands used the installed
requirements and DATABASE_URL pointing at the isolated database.

| Command | Before | After |
| --- | --- | --- |
| `pytest -q` without DATABASE_URL | 22 passed, 3 skipped | Not rerun; database-backed run below includes these tests |
| `pytest -q` with DATABASE_URL | 25 passed, 0 failed, 0 skipped (1.02s) | 33 passed, 0 failed, 0 skipped (1.43s) |
| `python evaluation.py --method keyword` | Hit@1/3 and MRR@3: 0.4286 (9/21) | 0.0476 (1/21) |
| `python evaluation_robust.py --method bm25` | Test Hit@1 .6, Hit@3 .7, MRR@3 .65; no-answer rejection 5/6 | Same metrics |
| `python evaluation.py --method semantic` | No new live baseline | Exit 1: `RuntimeError: GEMINI_API_KEY must be configured` |
| `python evaluation_robust.py --method semantic` | No new live baseline | Exit 1: `RuntimeError: GEMINI_API_KEY must be configured` |

The existing development-selection algorithm reports a BM25 cutoff of
3.93991886 before and 4.01479266 after because the input text changes. These are
measurement outputs; no application cutoff was changed. Do not interpret them
as new calibration recommendations.

The keyword decline exposes the existing first-substring baseline: the shared
`Service:` prefix itself matches the first corpus record. That baseline was
left unchanged for review. It should not be presented as semantic retrieval
quality or a production regression.

Eight new test cases pass: three API/evaluation parity variants using real
pgvector (structured/redacted, legacy, and no-match), one BM25 input check, and
four invalid-payload validation checks. Parity covers both semantic evaluators,
exact query text, ranking, scores and thresholded empty results. Gemini embeddings
are deterministic stubs and generation is stubbed: these tests prove processing
and retrieval plumbing parity, not live Gemini accuracy.

Both full-suite runs report the same two pre-existing warnings:
`StarletteDeprecationWarning` about httpx/TestClient and
`PytestUnknownMarkWarning` for the unregistered integration marker.

No root-cause experiments, threshold fixes, dependency fixes, or baseline-search
fixes were made. Stop here for review; nothing is merged into main.
