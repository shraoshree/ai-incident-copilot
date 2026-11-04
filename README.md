# AI Incident Copilot — Vector RAG Prototype

Backend service for incident triage using **FastAPI, PostgreSQL + pgvector, Gemini embeddings, and Gemini 3.6 Flash structured generation**. It retrieves relevant historical incidents, presents them to the model as untrusted reference context, validates output, and stores a lifecycle record in PostgreSQL.

> **Verification status (2026-11-04):** Live Gemini embeddings and pgvector retrieval were successfully exercised in GitHub Actions, including ingestion of all 21 synthetic incidents. Retrieval accuracy figures below are specific to a small, illustrative synthetic benchmark, not production or held-out performance. The full end-to-end request using Gemini 3.6 Flash generation, pgvector retrieval, and PostgreSQL storage also **passed** in GitHub Actions. See [successful live end-to-end run](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41853600620).

## Architecture

```text
POST /incident
  -> Pydantic request validation
  -> Gemini embedding (RETRIEVAL_QUERY)
  -> pgvector cosine similarity, top 3 + threshold
  -> historical context -> Gemini structured output
  -> retries/fallback on LLM failure
  -> PostgreSQL incident record + retrieved match metadata
  -> JSON response
```

Document ingestion: `knowledge/incidents.txt` -> Gemini embeddings (`RETRIEVAL_DOCUMENT`) -> `knowledge` table. The same model and dimension must be used for queries and documents. Historical records are illustrative synthetic engineering incidents, **not actual company records**.

## Quick start

Requires Docker and a Gemini API key with access to the embedding and generation models. No keys should be committed.

```bash
cp .env.example .env
# edit .env, set GEMINI_API_KEY
docker compose up -d db
# local Python virtual environment, then:
pip install -r requirements.txt
python retrieval.py init
python retrieval.py ingest
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs`. Or run the API through Compose:

```bash
docker compose up --build -d
docker compose exec api python retrieval.py init
docker compose exec api python retrieval.py ingest
```

Note: if both local Python and the API Compose service are running, use appropriate hostnames: local `DATABASE_URL` points to `localhost`, Compose API overrides it to `db`. `docker compose down -v` deletes database data.

### Example

```bash
curl -X POST http://localhost:8000/incident \
  -H 'Content-Type: application/json' \
  -d '{"service":"payments","alert":"High API latency","severity":"critical","description":"p99 response time rose above 2800ms"}'
```

Response includes `incident_id`, `analysis_source` (Gemini or fallback), `retrieval_matches` with scores, and `retrieval_ms`. This is a measured wall-time for embedding + retrieval, **not the total API latency**.

## APIs

| Endpoint | Description |
| --- | --- |
| `GET /` | Service identity |
| `GET /health` | Database-backed health |
| `POST /incident` | Retrieve, analyze and persist incident |
| `GET /incidents?status=open&limit=100` | Paginated-size capped incident list (limit, not cursor pagination) |
| `GET /incidents/{id}` | Fetch incident |
| `PATCH /incidents/{id}/status` | Enforce open → investigating/resolved → resolved |
| `GET /stats` | Aggregate status and critical counts |
| `GET /knowledge` | List knowledge (demo-only endpoint) |

## Evaluation

The labeled demo set `evaluation/cases.json` contains 21 paraphrased queries with an expected historical incident title. The matching knowledge corpus has 21 synthetic cases. The tests measure retrieval selection, not root-cause correctness.

```bash
python evaluation.py --method keyword
python evaluation.py --method semantic
pytest -q
```

**Verified live retrieval benchmark (21 labeled paraphrase cases, GitHub Actions 2026-11-04):**

| Retriever | Hit@1 | Hit@3 | MRR@3 |
|---|---:|---:|---:|
| Original keyword (first matching block) | 0.4286 | 0.4286 | 0.4286 |
| Gemini embeddings + pgvector top-3 | **1.0000** | **1.0000** | **1.0000** |

That is 9/21 vs 21/21 correct at rank 1, an improvement of **57.14 percentage points** on this limited synthetic benchmark. The examples are constructed from the same incident scenarios and are not a held-out real-world test; these scores cannot establish production root-cause accuracy. [Live evaluation run](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41851666221).

**CI verification:** 12 unit + PostgreSQL/pgvector integration tests passed on GitHub Actions (Nov 4, 2026), including real vector ranking and incident lifecycle tests using deterministic embeddings.

**Live evaluation workflow:** `.github/workflows/live-evaluation.yml` runs on a workflow-file change to this feature branch or manual dispatch once available on the default branch. It uses the existing GitHub repository secret (`GEMINI_API_KEY`), provisions a temporary pgvector database, and runs the keyword/semantic benchmarks and an API smoke test. Do not enable paid Gemini billing; if the free tier is exhausted, stop rather than upgrade. Never paste API keys into issues, commits or chat.

Metrics: Hit@1, Hit@3, MRR@3; the keyword baseline returns only the **first** match, mirroring the original implementation. These comparisons are for development only: tuning `MIN_SIMILARITY` on the same 21 examples would contaminate evaluation. Create an independent, held-out test set before claiming an improvement. Negative queries and abstention accuracy also need dedicated evaluation.

## Reliability and security considerations

- Gemini generation has a 15-second request timeout (configured through the SDK), up to 3 server-error attempts, and a conservative fallback. No fixed guarantee of all network failure recovery.
- Missing or failed semantic retrieval yields explicit unavailable context; it does **not** falsely claim a matching past incident.
- If database persistence fails, `POST /incident` returns HTTP 503 instead of falsely claiming the incident was saved.
- PostgreSQL transactions protect status updates with `FOR UPDATE`, avoiding the JSON file's lost-update risk.
- The knowledge corpus and incoming incident fields are **untrusted data**, not authoritative instructions for the model. Users must verify suggested actions with live evidence.
- The demo Compose database password is **development-only**. Never expose it or the demo API directly to the public internet. Authentication, access controls, redaction of secrets in incident descriptions, rate limiting, and incident-level audit logs are required before real production use.
- `/knowledge` exposes sample historical context and must be restricted or removed for sensitive deployments.
- An HNSW index is intentionally omitted at this tiny corpus size. Add/benchmark an index when scale justifies it.

## Project layout

```text
main.py                 FastAPI, validated LLM analysis, lifecycle endpoints
database.py             pgvector and incident PostgreSQL schema
retrieval.py            embedding, ingestion, vector search, keyword baseline
evaluation.py           labeled retrieval benchmark
knowledge/incidents.txt 21 synthetic incident records
evaluation/cases.json   21 illustrative paraphrases and labels
test_main.py            mock-based unit/API tests
Dockerfile              application image
docker-compose.yml      local pgvector database + application
.github/workflows/      unit-test CI
```

### Optional next steps beyond the portfolio milestone

Unit, database integration, live retrieval, and end-to-end Gemini generation were exercised successfully in GitHub Actions. **Still required before calling this production-ready:** a held-out real-world labeled dataset, negative-query abstention tests, threshold calibration on a separate development set, load testing, authenticated access, and data-redaction controls. The current benchmark is a meaningful functional check, not proof of reliability in production. Do not call the application production-ready before authentication, secrets redaction, and access control exist.
