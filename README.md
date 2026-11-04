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

I compared the original keyword approach with semantic retrieval using 21 synthetic historical incidents and 21 labeled paraphrased queries. Each query had one expected historical incident.

| Retriever | Hit@1 | Hit@3 | MRR@3 |
| --- | ---: | ---: | ---: |
| First-match keyword search | 42.86% | 42.86% | 42.86% |
| Gemini embeddings + pgvector | **100%** | **100%** | **100%** |

On this dataset, the correct historical incident was ranked first for 9 of 21 keyword queries and 21 of 21 semantic queries. That is a 57.14 percentage-point increase in Hit@1.

These numbers measure **retrieval on a small, synthetic benchmark**, not root-cause diagnosis accuracy or performance on real company incidents. A larger held-out test set, including irrelevant queries, is needed to judge generalization and abstention.

Test evidence:

- [Live retrieval evaluation and full API smoke test](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41853600620)
- [Automated tests](https://github.com/shraoshree/ai-incident-copilot/actions/runs/41852602078) - 12 tests passed with PostgreSQL and pgvector.

The live smoke test exercised the complete path: Gemini embedding, pgvector retrieval, Gemini 3.6 Flash generation, and PostgreSQL persistence.

## Run locally

Requirements: Python 3.12, Docker, and a Gemini API key with access to the embedding and generation models.

1. Copy the environment example and add your API key to the local `.env` file. Do not commit it.

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

Open `http://localhost:8000/docs` to try the endpoints through FastAPI's interactive documentation.

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
evaluation/cases.json      21 labeled evaluation queries
test_main.py               Unit and API tests
test_integration.py        PostgreSQL and pgvector integration tests
docker-compose.yml         Local database and API containers
.github/workflows/         CI tests and optional live evaluation
```

## Scope and next steps

This is a locally runnable portfolio prototype, not a production incident-management platform. The incident corpus is synthetic, and the API does not yet include authentication, sensitive-data redaction, or rate limiting. The demo database password is for local development only. Do not expose this configuration publicly or use it to store real incident details.

The next useful improvements would be a held-out evaluation set with irrelevant queries, threshold calibration, access controls, and more realistic incident records. Those are separate from the working portfolio milestone.
