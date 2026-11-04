# AI Incident Copilot

A FastAPI service that helps investigate production incidents by finding similar past incidents and using them as context for an AI-generated troubleshooting plan.

I built this to explore a problem I worked around in production support: when an alert fires, the first challenge is often finding the right logs, previous incidents, and investigation steps quickly. The goal here is to make that starting point easier, not to replace an engineer's judgment.

## What it does

- Accepts an incident with a service, alert, severity, and description.
- Searches a PostgreSQL knowledge base for relevant historical incidents using Gemini embeddings and pgvector.
- Passes the retrieved context to Gemini 3.6 Flash to suggest a likely cause, investigation steps, and an immediate action.
- Saves the incident, retrieval matches, analysis source, and status in PostgreSQL.
- Supports listing incidents, viewing statistics, and updating status from open to investigating to resolved.
- Returns a conservative fallback analysis if the generation service is unavailable.
- Requires an API key for incident endpoints, limits requests per client, and masks common secrets before retrieval, AI generation, and storage.

## Why I changed the retrieval approach

The first version used keyword matching. It looked for words from the incoming alert inside historical incident text and returned the first match. That worked for similar wording but was unreliable when two descriptions meant the same thing using different terms.

For example, an alert about slow checkout requests may be related to a past incident about database connection pool exhaustion, even if the wording does not overlap much.

I replaced that lookup with embeddings and pgvector:

1. Convert each historical incident into a 768-dimensional embedding and store it in PostgreSQL.
2. Embed the incoming incident description using the same embedding model.
3. Search for the nearest historical incidents using cosine similarity.
4. Keep up to three matches above a configurable similarity threshold.
5. Give the matches to Gemini as reference material, then save the generated analysis.

The threshold matters because a nearest match is not necessarily a useful match. When nothing meets the threshold, the application tells the model that no sufficiently similar historical incident was found.

## Architecture

```text
                     Incident request
                            |
                      FastAPI + Pydantic
                            |
                   Gemini query embedding
                            |
                PostgreSQL + pgvector search
                            |
                   Top 3 relevant incidents
                            |
                Gemini 3.6 Flash analysis
                            |
             PostgreSQL incident + analysis
                            |
                        JSON response

Knowledge ingestion:
knowledge/incidents.txt -> Gemini document embeddings -> pgvector
```

**Main technologies:** Python, FastAPI, Pydantic, PostgreSQL, pgvector, Google Gemini API, pytest, Docker Compose, and GitHub Actions.

## Evaluation

The first retrieval benchmark used 21 synthetic historical incidents and 21 labeled paraphrases. Because the incidents and paraphrases were created together, the 100% result is **not** an independent estimate of accuracy. Some queries reuse incident terminology. I keep that original measurement for comparison, but now also test abstention and a stronger lexical baseline.

### New benchmark: BM25 vs semantic retrieval

I added 12 queries for which **none** of the 21 historical incidents is an appropriate answer. The benchmark has 33 queries total: 21 answerable and 12 with no answer. Each group is split by alternating index into a development set (11 answerable, 6 no-answer) and a test set (10 answerable, 6 no-answer).

Both methods retrieve their top three candidates. The BM25 baseline uses Okapi BM25 scoring with token frequencies and document-length normalization, not the old first-substring-match approach. Each method gets its **own** cutoff, selected using only the development set. The objective is balanced decision accuracy: the average of correct top-1 positive answers and correct rejections of no-answer queries. Ties favor the lower threshold. The test set is not used to select the cutoff.

**Live test results - November 4, 2026**

| Held-out test metric | Gemini embeddings + pgvector | BM25 |
| --- | ---: | ---: |
| Positive queries | 10 | 10 |
| No-answer queries | 6 | 6 |
| Hit@1, on positives | **100%** | 60% |
| Hit@3, on positives | **100%** | 70% |
| MRR@3, on positives | **100%** | 65% |
| No-answer abstention | 83.33% (5/6) | 83.33% (5/6) |
| False acceptance of no-answer queries | 16.67% (1/6) | 16.67% (1/6) |
| Development-selected threshold | 0.61235 cosine similarity | 3.82436 BM25 score |

The semantic retriever ranked the correct incident first for all 10 answerable test queries, but it also **incorrectly accepted 1 of 6 irrelevant queries**. That is a real limitation worth addressing rather than hiding behind the 100% positive-retrieval result.

### What the similarity scores tell us

Cosine similarity is reported on the semantic results. The test distributions are:

| Top retrieval score distribution | Relevant incident matched in top 3 (10 queries) | Highest score for irrelevant query (6 queries) |
| --- | ---: | ---: |
| Minimum | 0.6981 | 0.5363 |
| Median | 0.7261 | 0.5618 |
| Maximum | 0.7662 | 0.6262 |

The development-selected cutoff was **0.61235**. One irrelevant test query scored **0.6262**, crossing the cutoff and producing a false acceptance. This shows why a threshold calibrated on a small set cannot be treated as universally reliable. BM25 values are different scoring units, so they should not be compared directly with cosine similarities.

**Important:** The application currently defaults to `MIN_SIMILARITY=0.50`. The 0.61235 threshold is an **evaluation result**, not a silently deployed production setting. It needs validation on independent data before being adopted for real use.

### Previous baseline and metric interpretation

The original 21-query benchmark reported 42.86% for first-match keyword lookup and 100% for semantic search at Hit@1, Hit@3 and MRR@3. Those matching values are expected: keyword lookup returns at most one result, and all correct semantic matches happened at rank 1. The new BM25 results include hits at ranks 2 and 3, so its metrics differ.

**Reproduce results:**

```bash
python evaluation_robust.py --method bm25
python evaluation_robust.py --method semantic
```

BM25 requires no Gemini calls. Semantic evaluation uses the configured embedding API and ingested pgvector database; remain within the free-tier limits.

- [Committed evaluation results](evaluation/results_2026-11-04.json) - preserved in the repository even if GitHub Actions logs expire.
- [Live semantic and BM25 benchmark workflow](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41857655455)
- [Automated evaluation and integration tests](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41857795046)

These are synthetic scenarios, created for this project. They are split by **query**, not by incident or author. The test queries share the same 21 incident concepts used for development; the benchmark is not independently authored, does not measure root-cause diagnosis accuracy, and remains too small to claim production-grade abstention.

## Run locally

Requirements: Python 3.12, Docker, and a Gemini API key with access to the embedding and generation models.

1. Copy the environment example and add your Gemini API key and a separate, randomly generated `COPILOT_API_KEY` to the local `.env` file. Do not commit either key.

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use the printed token as `COPILOT_API_KEY`. It is different from your Gemini API key.

```bash
cp .env.example .env
```

2. Start PostgreSQL with pgvector.

```bash
docker compose up -d db
```

3. Install Python dependencies, create the database tables, and ingest the demo incidents.

```bash
pip install -r requirements.txt
python retrieval.py init
python retrieval.py ingest
```

4. Start the API.

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000/docs` to try the endpoints through FastAPI's interactive documentation. Click **Authorize** and enter the value of `COPILOT_API_KEY` before calling protected endpoints. The `/` and `/health` endpoints remain accessible without a key.

Alternatively, run both the API and database in Docker:

```bash
docker compose up --build -d
docker compose exec api python retrieval.py init
docker compose exec api python retrieval.py ingest
```

The `.env.example` database URL uses `localhost` for running Python on your computer. Docker Compose overrides it with the database service hostname when running the API in a container.

### Try an incident

```bash
curl -X POST http://localhost:8000/incident \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $COPILOT_API_KEY" \
  -d '{"service":"payments","alert":"High API latency","severity":"critical","description":"p99 checkout latency increased above 2800ms"}'
```

The response contains the incident ID, retrieved knowledge matches and similarity scores, structured analysis, and its source (`gemini` or `fallback`). It also records retrieval time in milliseconds. This is not total API latency.

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /incident` | Retrieve context, analyze, and save a new incident |
| `GET /health` | Check service and database connectivity |
| `GET /incidents` | List incidents, optionally filtered by status |
| `GET /incidents/{incident_id}` | Get a saved incident |
| `PATCH /incidents/{incident_id}/status` | Move an incident through its allowed status transitions |
| `GET /stats` | View incident counts by status and critical severity |
| `GET /knowledge` | Inspect the sample knowledge base |

## Production-readiness safeguards

- **API authentication:** All incident, stats, and knowledge endpoints require the `X-API-Key` header. If `COPILOT_API_KEY` is missing, protected routes fail closed. This is a single shared demo key, not user accounts or role-based permissions.
- **Rate limiting:** An in-process sliding-window limiter permits 20 incident submissions per client IP per minute and 120 other authenticated requests per minute. Excess traffic receives HTTP 429 and `Retry-After`. These limits reset on restart and are not coordinated across workers or replicas.
- **Sensitive-data masking:** Before embedding an incoming incident, sending it to Gemini, or persisting it, the API masks common credential patterns, bearer tokens, email addresses, and IPv4 addresses. Regex-based redaction cannot catch every secret or personally identifiable value. Never submit real confidential incident data without a stronger data-classification and redaction pipeline.
- **Payload cap:** Requests with a declared content length above 32 KiB are rejected. This is a basic safeguard, not a complete streaming body-size defense.
- **API documentation:** Swagger is public locally, but its incident operations require authorization.

These controls improve the local portfolio prototype. Internet-facing production use would still require TLS, per-user identity and authorization, a shared rate limiter, audited key rotation, stronger privacy controls, and operational monitoring.

## Load testing (no Gemini calls)

The included load test calls **only** `GET /health`, so running it does not invoke Gemini or use embedding tokens:

```bash
python load_test.py --url http://localhost:8000 --requests 100 --concurrency 10
```

It reports throughput, success rate, average latency, and p50/p95/p99 latency. The benchmark runs against your own machine; numbers depend on local hardware and database state. GitHub CI also runs 100 health requests with concurrency 10. In one GitHub Actions run, **100/100 requests succeeded**, with about **67.75 requests/second** and **214.55 ms p95 latency**. These are measurements of the database-backed health endpoint on a shared CI runner, not the Gemini analysis endpoint or a production performance guarantee. [Load-test run](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41856010469).

## Testing

```bash
pytest -q
python evaluation.py --method keyword
python evaluation.py --method semantic
```

The semantic evaluation requires a configured database containing ingested knowledge and access to Gemini embeddings. The unit and database integration tests use deterministic embeddings, so they do not need to call Gemini.

GitHub Actions also runs the unit and pgvector integration tests. A separate live workflow exercises embedding ingestion, retrieval evaluation, and the full incident request using the repository's API key secret.

## Reliability decisions

- **Structured responses:** Pydantic defines the expected fields for both requests and generated analysis.
- **Retries and fallback:** Gemini generation retries server errors up to three times. If it remains unavailable, the API returns a clearly labeled fallback instead of pretending the analysis came from Gemini.
- **Durable storage:** Incident records and status changes are kept in PostgreSQL rather than a local JSON file. Status updates use a row lock to avoid conflicting transitions.
- **Honest failure reporting:** If an incident cannot be saved, the API returns HTTP 503 rather than reporting success.
- **Grounded context:** Retrieved incident text is treated as reference data, not instructions. Suggested actions still require verification against current logs and metrics.

## Project structure

```text
main.py                    API routes and incident analysis
database.py                PostgreSQL schema and connection helpers
retrieval.py               Embeddings, ingestion, semantic and keyword retrieval
evaluation.py              Retrieval benchmark and metrics
knowledge/incidents.txt    21 synthetic historical incidents
evaluation/cases.json      21 labeled positive evaluation queries
 evaluation/negative_cases.json 12 queries with no relevant incident
 evaluation/results_2026-11-04.json Archived benchmark evidence
 evaluation_robust.py        BM25/semantic retrieval and abstention evaluation
test_main.py               Unit and API tests
test_integration.py        PostgreSQL and pgvector integration tests
 test_evaluation_robust.py   BM25, split and threshold evaluation tests
docker-compose.yml         Local database and API containers
.github/workflows/         CI tests and optional live evaluation
```

## Scope and next steps

This is a locally runnable portfolio prototype, not a production incident-management platform. The incident corpus is synthetic. The project includes basic shared-key authentication, regex-based sensitive-data redaction, and an in-process rate limiter, but these are not substitutes for production identity management and privacy controls. The demo database password is for local development only. Do not expose this configuration publicly or use it to store real incident details.

The next useful improvements would be a held-out evaluation set with irrelevant queries, threshold calibration, per-user access controls, stronger redaction, a multi-instance rate limiter, and more realistic incident records. Those are separate from the working portfolio milestone.
